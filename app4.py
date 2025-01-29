import doublemetaphone
import streamlit as st
import pinecone
import openai
from difflib import SequenceMatcher
import os

# Initialize API keys
from dotenv import load_dotenv
load_dotenv()

# Initialize OpenAI client
openai.api_key = os.environ['OPENAI_API_KEY']

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=os.environ['PINECONE_API_KEY'])

try:
    wordmark_index = pc.Index("wordmark-index")
    phonetic_index = pc.Index("phonetic-index")
    class_index = pc.Index("index-all")
except Exception as e:
    st.error(f"Failed to connect to Pinecone: {str(e)}")

def get_phonetic_representation(word):
    """
    Generate a phonetic representation for a word using Double Metaphone.
    If both primary and secondary values are None, fallback to the original word.
    """
    primary, secondary = doublemetaphone.doublemetaphone(word)
    return primary or secondary or word

def get_embedding(text, model="text-embedding-ada-002"):
    """
    Generate an embedding for the input text using OpenAI's embedding model.
    """
    try:
        response = openai.Embedding.create(
            model=model,
            input=[text]
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Error generating embedding: {str(e)}")
        return None

def calculate_phonetic_similarity(word1, word2):
    """
    Calculate phonetic similarity between two words using Double Metaphone.
    Returns a score between 0.0 and 1.0 based on similarity.
    """
    phonetic1 = get_phonetic_representation(word1)
    phonetic2 = get_phonetic_representation(word2)
    
    # Use SequenceMatcher to calculate similarity between phonetic representations
    similarity = SequenceMatcher(None, phonetic1, phonetic2).ratio()
    return similarity

def calculate_hybrid_score(phonetic_score, semantic_score, phonetic_weight=0.6, semantic_weight=0.4):
    """
    Calculate a hybrid score using a weighted average of phonetic and semantic similarity.
    """
    return (phonetic_weight * phonetic_score) + (semantic_weight * semantic_score)

def check_multiple_phonetic_matches(input_wordMark, input_class, index, model="text-embedding-ada-002", namespace="default"):
    """
    Check for matches based on both phonetic and semantic similarity.
    """
    try:
        # Step 1: Get phonetic representation
        phonetic_representation = get_phonetic_representation(input_wordMark)
        
        # Step 2: Get semantic embedding
        input_embedding = get_embedding(input_wordMark, model)
        
        if input_embedding is None:
            st.error("Could not generate embedding for input")
            return None

        # Step 3: Query Pinecone
        query_result = index.query(
            vector=input_embedding,
            top_k=5,  # Increase top_k to get more results
            include_metadata=True,
            namespace=namespace
        )

        # Step 4: Calculate hybrid scores
        matches = []
        for match in query_result.matches:
            stored_wordMark = match.metadata.get("wordMark", "")
            stored_classes = match.metadata.get("wclass", [])
            stored_phonetic = match.metadata.get("Phonetic_Representation", "")

            # Calculate phonetic similarity
            phonetic_score = calculate_phonetic_similarity(input_wordMark, stored_wordMark)
            
            # Get semantic similarity (from Pinecone query)
            semantic_score = match.score

            # Calculate hybrid score
            hybrid_score = calculate_hybrid_score(phonetic_score, semantic_score)

            # Add match to results
            matches.append({
                "Matching Wordmark": stored_wordMark,
                "Phonetic Representation": stored_phonetic,
                "Class": stored_classes,
                "Phonetic Score": phonetic_score,
                "Semantic Score": semantic_score,
                "Hybrid Score": hybrid_score
            })

        # Step 5: Sort matches by hybrid score
        matches = sorted(matches, key=lambda x: x["Hybrid Score"], reverse=True)

        return matches
    except Exception as e:
        st.error(f"Error checking phonetic matches: {str(e)}")
        return None

def suggest_similar_names(input_wordMark):
    """
    Suggest alternative names using OpenAI's GPT-4.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant who generates creative and unique alternative names for businesses."
                },
                {
                    "role": "user",
                    "content": f"Suggest five creative and unique alternative names for the word '{input_wordMark}'."
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

def main():
    st.title("Trademark Namesearch")

    st.write("This tool helps you find similar/phonetic matches for wordmarks and suggests alternative names.")

    col1, col2 = st.columns(2)
    
    with col1:
        input_wordMark = st.text_input("Enter the wordMark:", "")
    with col2:
        input_class = st.text_input("Enter the class:", "")

    if st.button("Search"):
        if not input_wordMark or not input_class:
            st.warning("Please enter both wordMark and class.")
            return

        with st.spinner("Searching for matches..."):
            matches = check_multiple_phonetic_matches(input_wordMark, input_class, class_index)

        if matches:
            st.success(f"Found matches!")
            
            # Filter matches based on hybrid score threshold
            filtered_matches = [match for match in matches if match["Hybrid Score"] > 0.8]
            
            if filtered_matches:
                for match in filtered_matches:
                    with st.expander(f"Match: {match['Matching Wordmark']}"):
                        st.write(f"- Phonetic Representation: {match['Phonetic Representation']}")
                        st.write(f"- Class: {match['Class']}")
                        st.write(f"- Phonetic Score: {match['Phonetic Score']:.2f}")
                        st.write(f"- Semantic Score: {match['Semantic Score']:.2f}")
                        st.write(f"- Hybrid Score: {match['Hybrid Score']:.2f}")

                        if str(input_class) in match['Class']:
                            st.write("\n**Generating alternative suggestions...**")
                            unique_suggestions = get_unique_suggestions(match['Matching Wordmark'], [wordmark_index, phonetic_index, class_index])
                            
                            if unique_suggestions:
                                st.write("**Unique Suggestions:**")
                                for suggestion in unique_suggestions:
                                    st.write(f"- {suggestion}")
                            else:
                                st.info("All generated suggestions already exist in the database.")
            else:
                st.info("No matches found with a hybrid score above 0.8. You can proceed with the name as no similar names exist.")
        else:
            st.info("No matches found for the given wordMark and class.")

if __name__ == "__main__":
    main()