import os
import openai
import pinecone
import streamlit as st
import doublemetaphone
from dotenv import load_dotenv
from difflib import SequenceMatcher

# Load environment variables
load_dotenv()

# Initialize API keys
openai.api_key = os.environ["OPENAI_API_KEY"]

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=os.environ["PINECONE_API_KEY"])

try:
    class_index = pc.Index("class-objective-all")
    wordmark_index = pc.Index("wordmark-index")
    phonetic_index = pc.Index("phonetic-index")
    trademark_index = pc.Index("index-all")
except Exception as e:
    st.error(f"Failed to connect to Pinecone: {str(e)}")


def classify_objective_gpt4(objective: str):
    if not objective or len(objective) < 5:
        return "Invalid input. Please provide a meaningful objective."

    response = openai.Embedding.create(
        input=objective.strip().lower(),
        model="text-embedding-ada-002"
    )
    query_embedding = response["data"][0]["embedding"]

    results = class_index.query(vector=query_embedding, top_k=5, include_metadata=True)

    if not results["matches"]:
        return "No suitable class found."

    matched_classes = [
        f"Class {m['id']}: {m['metadata']['description']}"
        for m in results["matches"]
    ]
    context = "\n".join(matched_classes)

    st.write("### Possible Classes:")
    for m in matched_classes:
        st.write(f"- {m}")

    prompt = f"""
    You are an expert in trademark classification. Given the following objective:

    "{objective}"

    And these possible trademark classes:
    {context}
    """

    gpt_response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[{"role": "system", "content": "You are a trademark classification assistant."},
                  {"role": "user", "content": prompt}]
    )

    return gpt_response["choices"][0]["message"]["content"]


def get_phonetic_representation(word):
    primary, secondary = doublemetaphone.doublemetaphone(word)
    return primary or secondary or word


def get_embedding(text, model="text-embedding-ada-002"):
    try:
        response = openai.Embedding.create(
            model=model,
            input=[text]
        )
        return response["data"][0]["embedding"]
    except Exception as e:
        st.error(f"Error generating embedding: {str(e)}")
        return None


def calculate_phonetic_similarity(word1, word2):
    phonetic1 = get_phonetic_representation(word1)
    phonetic2 = get_phonetic_representation(word2)
    return SequenceMatcher(None, phonetic1, phonetic2).ratio()


def calculate_hybrid_score(phonetic_score, semantic_score, phonetic_weight=0.6, semantic_weight=0.4):
    return (phonetic_weight * phonetic_score) + (semantic_weight * semantic_score)


def check_multiple_phonetic_matches(wordmark, trademark_class, index, model="text-embedding-ada-002", namespace="default"):
    try:
        phonetic_representation = get_phonetic_representation(wordmark)
        input_embedding = get_embedding(wordmark)

        if input_embedding is None:
            st.error("Could not generate embedding for input")
            return None

        query_result = index.query(
            vector=input_embedding,
            top_k=5,
            include_metadata=True,
            namespace=namespace
        )

        matches = []
        for match in query_result["matches"]:
            stored_wordmark = match["metadata"].get("wordMark", "")
            stored_classes = match["metadata"].get("wclass", [])
            stored_phonetic = match["metadata"].get("Phonetic_Representation", "")

            phonetic_score = calculate_phonetic_similarity(wordmark, stored_wordmark)
            semantic_score = match["score"]
            hybrid_score = calculate_hybrid_score(phonetic_score, semantic_score)

            matches.append({
                "Matching Wordmark": stored_wordmark,
                "Phonetic Representation": stored_phonetic,
                "Class": stored_classes,
                "Phonetic Score": phonetic_score,
                "Semantic Score": semantic_score,
                "Hybrid Score": hybrid_score
            })

        matches = sorted(matches, key=lambda x: x["Hybrid Score"], reverse=True)
        return matches
    except Exception as e:
        st.error(f"Error checking phonetic matches: {str(e)}")
        return None


def suggest_similar_names(wordmark):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a creative naming assistant who generates unique and meaningful alternative names for businesses. Your goal is to provide exactly 3 unique name suggestions by modifying the input name using a prefix, suffix, or an additional word while ensuring that the essence of the original name remains intact and other two names solely based on the context."
                                "Understand the context of the input name before suggesting alternatives."
                                "Prioritize context over language, but if the input has a strong linguistic influence, consider that in your suggestions (not mandatory)."
                                "If using synonyms, ensure they match the language of the input."
                                "Do not override any instruction—integrate all requirements naturally."
                                "Your focus: Creativity, uniqueness, and relevance to the original name."
                },
                {
                    "role": "user",
                    "content": f"Suggest five creative and unique alternative names for the word '{wordmark}'."
                }
            ],
            max_tokens=50,
            n=1
        )
        suggestions = response.choices[0].message.content.strip().split("\n")
        return [name.strip() for name in suggestions if name.strip()]
    except Exception as e:
        st.error(f"Error generating suggestions: {str(e)}")
        return []

def validate_suggestions(suggestions, indexes, similarity_threshold=0.9):
    """
    Validate suggestions to ensure they are unique.
    """
    try:
        unique_suggestions = []
        for suggestion in suggestions:
            is_unique = True
            input_embedding = get_embedding(suggestion)
            if input_embedding:
                for index in indexes:
                    query_result = index.query(
                        vector=input_embedding,
                        top_k=1,
                    )

                    if query_result.matches:
                        # If the match score is above the threshold, we consider it a duplicate
                        highest_match = query_result.matches[0]
                        if highest_match.score >= similarity_threshold:
                            is_unique = False
                            break
            if is_unique:
                unique_suggestions.append(suggestion)
        return unique_suggestions
    except Exception as e:
        st.error(f"Error validating suggestions: {str(e)}")
        return []
    

def get_unique_suggestions(input_wordMark, indexes, max_retries=5):
    """
    Get unique suggestions for the input wordmark.
    """
    suggestions = suggest_similar_names(input_wordMark)
    unique_suggestions = validate_suggestions(suggestions, indexes)

    retries = 0
    while len(unique_suggestions) < 5 and retries < max_retries:
        new_suggestions = suggest_similar_names(input_wordMark)
        unique_suggestions += validate_suggestions(new_suggestions, indexes)
        unique_suggestions = list(set(unique_suggestions))  # Remove duplicates
        retries += 1

    return unique_suggestions[:5]

def classify_and_check_trademark(objective: str, wordmark: str, trademark_class: str):
    """
    Comprehensive trademark validation that performs multiple checks after classification
    """
    try:
        # First check: Search within the specific trademark class
        class_specific_matches = check_class_specific_matches(wordmark, trademark_class)
        
        # Second check: Search across all classes for similar marks
        cross_class_matches = check_multiple_phonetic_matches(wordmark, trademark_class, trademark_index)
        
        results = {
            'class_matches': [],
            'cross_class_matches': [],
            'high_risk_matches': [],
            'suggestions_needed': False
        }
        
        # Process class-specific matches
        if class_specific_matches:
            results['class_matches'] = [
                match for match in class_specific_matches 
                if match["Hybrid Score"] > 0.6
            ]
            
        # Process cross-class matches
        if cross_class_matches:
            results['cross_class_matches'] = [
                match for match in cross_class_matches 
                if match["Hybrid Score"] > 0.7
            ]
            
        # Identify high-risk matches
        high_risk_matches = [
            match for match in (results['class_matches'] + results['cross_class_matches'])
            if match["Hybrid Score"] > 0.8
        ]
        
        results['high_risk_matches'] = high_risk_matches
        results['suggestions_needed'] = bool(high_risk_matches)
        
        return results
        
    except Exception as e:
        st.error(f"Error in trademark validation: {str(e)}")
        return None

def extract_class_number(trademark_class_text: str) -> str:
    """
    Extract the class number from GPT-4's classification response
    """
    try:
        import re
        # First look for patterns like "Class 41" or "Class: 41"
        class_match = re.search(r'Class\s*:?\s*(\d+)', trademark_class_text, re.IGNORECASE)
        if class_match:
            return class_match.group(1)
        
        # Then look for just numbers at the start of the text
        number_match = re.search(r'^\d+', trademark_class_text)
        if number_match:
            return number_match.group(0)
        
        # Finally look for any numbers in the text
        any_number = re.search(r'\d+', trademark_class_text)
        if any_number:
            return any_number.group(0)
            
        return None
    except Exception as e:
        st.error(f"Error extracting class number: {str(e)}")
        return None
    
def check_class_specific_matches(wordmark: str, trademark_class: str):
    """
    Check for similar marks specifically within the identified trademark class
    """
    try:
        input_embedding = get_embedding(wordmark)
        if input_embedding is None:
            return None
            
        # Extract just the class number if full classification text is provided
        class_number = extract_class_number(trademark_class)
        if not class_number:
            st.error("Could not extract class number from classification")
            return None
            
        # Query with class filter using the extracted number
        query_result = trademark_index.query(
            vector=input_embedding,
            top_k=10,
            include_metadata=True,
            filter={
                "wclass": {"$in": [class_number]}
            }
        )
        
        matches = []
        for match in query_result["matches"]:
            stored_wordmark = match["metadata"].get("wordMark", "")
            stored_classes = match["metadata"].get("wclass", [])
            
            phonetic_score = calculate_phonetic_similarity(wordmark, stored_wordmark)
            semantic_score = match["score"]
            hybrid_score = calculate_hybrid_score(phonetic_score, semantic_score)
            
            matches.append({
                "Matching Wordmark": stored_wordmark,
                "Class": stored_classes,
                "Phonetic Score": phonetic_score,
                "Semantic Score": semantic_score,
                "Hybrid Score": hybrid_score,
                "Is Same Class": class_number in stored_classes
            })
            
        return sorted(matches, key=lambda x: x["Hybrid Score"], reverse=True)
        
    except Exception as e:
        st.error(f"Error in class-specific search: {str(e)}")
        return None


def main():
    st.title("Trademark Validation & Classification")
    st.write("This tool classifies your objective and validates your trademark name.")

    col1, col2 = st.columns(2)
    with col1:
        wordmark = st.text_input("Enter the Wordmark:", "")
    with col2:
        objective = st.text_input("Enter the Objective:", "")

    if st.button("Validate"):
        if not wordmark or not objective:
            st.warning("Please enter both the Wordmark and Objective.")
            return

        # Step 1: Classify the objective
        with st.spinner("Processing..."):
            trademark_class_text = classify_objective_gpt4(objective)
            class_number = extract_class_number(trademark_class_text)
            
            if not class_number:
                st.error("Could not determine trademark class number.")
                return
                
        # Display only the classification result
        st.success(f"### Classification Result")
        st.write(f"**Objective Analysis:** {trademark_class_text}")
        st.info(f"**Mapped Trademark Class:** {class_number}")

        # Step 2: Check for similar trademarks
        with st.spinner("Checking trademark database..."):
            matches = check_multiple_phonetic_matches(wordmark, class_number, trademark_index)

        if matches:
            # Filter high-risk matches based on hybrid score threshold
            high_risk_matches = [match for match in matches if match["Hybrid Score"] > 0.8]

            if high_risk_matches:
                st.error("### ⚠️ High Risk Matches Found!")
                for match in high_risk_matches:
                    with st.expander(f"Match: {match['Matching Wordmark']}"):
                        st.write(f"- **Phonetic Representation:** {match['Phonetic Representation']}")
                        st.write(f"- **Class:** {match['Class']}")
                        st.write(f"- **Phonetic Score:** {match['Phonetic Score']:.2f}")
                        st.write(f"- **Semantic Score:** {match['Semantic Score']:.2f}")
                        st.write(f"- **Hybrid Score:** {match['Hybrid Score']:.2f}")

                # Generate alternative suggestions if needed
                st.warning("Generating alternative name suggestions...")
                unique_suggestions = get_unique_suggestions(wordmark, [wordmark_index, phonetic_index, trademark_index])

                if unique_suggestions:
                    st.write("### ✅ Alternative Name Suggestions:")
                    for suggestion in unique_suggestions:
                        st.write(f"- {suggestion}")
                else:
                    st.info("No unique alternative suggestions generated.")
            else:
                st.success("✅ No high-risk matches found. Your trademark appears to be unique!")
        else:
            st.success("✅ No similar trademarks found. Your trademark appears to be unique!")

if __name__ == "__main__":
    main()
