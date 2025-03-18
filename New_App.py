import pandas as pd
import numpy as np
import re
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import matplotlib.pyplot as plt
import networkx as nx
from gensim.models import Word2Vec
from collections import defaultdict
import streamlit as st

import nltk
nltk.download('punkt')  # Pastikan resource 'punkt' sudah diunduh sebelum digunakan

from nltk.tokenize import word_tokenize

# Fungsi untuk memproses review yang dimasukkan
def process_reviews(input_review):
    # Proses review pengguna
    review_df = pd.DataFrame({'review': [input_review]})
    review_df['review'] = review_df['review'].apply(preprocess_text)
    
    # Tokenisasi
    review_df["tokenized"] = review_df["review"].apply(lambda x: [word for word in word_tokenize(x) if len(word) > 1])

    return review_df


# Fungsi untuk melakukan preprocessing pada teks
def preprocess_text(text):
    factory = StemmerFactory()
    stopword_factory = StopWordRemoverFactory()
    stemmer = factory.create_stemmer()
    stopword_remover = stopword_factory.create_stop_word_remover()
    
    # Menghapus karakter duplikat
    text = re.sub(r'(.)\1+', r'\1', text)  
    text = stopword_remover.remove(text)   # Menghapus stopwords
    text = text.translate(str.maketrans('', '', string.punctuation))  # Menghapus tanda baca
    return text

# Fungsi untuk menampilkan grafik TF-IDF
def plot_tfidf(tfidf_df, document_index=0, top_n=10):
    tfidf_scores = tfidf_df.iloc[document_index]
    sorted_tfidf_scores = tfidf_scores.sort_values(ascending=False)
    top_tfidf_scores = sorted_tfidf_scores[:top_n]
    
    plt.figure(figsize=(10, 6))
    plt.bar(top_tfidf_scores.index, top_tfidf_scores.values)
    plt.xlabel("Words")
    plt.ylabel("TF-IDF Score")
    plt.title(f"Top {top_n} Words with Highest TF-IDF Scores for Document {document_index}")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(plt)

# Fungsi untuk membuat Word2Vec Graph
def GraphWord2Vec(text):
    w2v_model = Word2Vec([text], vector_size=64, window=2, min_count=1, sg=0)
    word_vectors = w2v_model.wv
    words_freq = defaultdict(int)
    
    for word in text:
        words_freq[word] += 1
    n_words = len(words_freq)
    word_idx = dict(zip(words_freq.keys(), range(n_words)))
    
    adj_matrix = np.zeros((n_words, n_words))
    for word_i in word_idx:
        for word_j in word_idx:
            if word_i != word_j:
                adj_matrix[word_idx[word_i], word_idx[word_j]] = word_vectors.similarity(word_i, word_j)
                
    graph = nx.Graph(adj_matrix)
    labels = {v: k for k, v in word_idx.items()}
    graph = nx.relabel_nodes(graph, labels)
    features = np.array([word_vectors[word] for word in text])
    
    return adj_matrix, features, graph

# Fungsi untuk melakukan perhitungan TF-IDF dan cosine similarity
def tfidf_similarity(review_df, expert_df):
    vectorizer = TfidfVectorizer()
    tfidf_matrix_reviews = vectorizer.fit_transform(review_df['review'])
    
    expert_df['processed_review'] = expert_df['review'].apply(lambda x: x.lower())
    tfidf_matrix_expert = vectorizer.transform(expert_df['processed_review'])
    
    cosine_similarities = cosine_similarity(tfidf_matrix_expert, tfidf_matrix_reviews)
    matched_indices = [cosine_similarities[i].argmax() for i in range(len(expert_df))]
    
    merged_df = pd.DataFrame()
    for i in range(len(expert_df)):
        expert_row = expert_df.iloc[i]
        review_row = review_df.iloc[matched_indices[i]]
        combined_row = pd.concat([expert_row, review_row])
        merged_df = pd.concat([merged_df, combined_row.to_frame().T], ignore_index=True)
    
    return merged_df

# Fungsi untuk memproses review yang dimasukkan
def process_reviews(input_review):
    # Proses review pengguna
    review_df = pd.DataFrame({'review': [input_review]})
    review_df['review'] = review_df['review'].apply(preprocess_text)
    
    # Tokenisasi
    nltk.download('punkt')
    from nltk.tokenize import word_tokenize
    review_df["tokenized"] = review_df["review"].apply(lambda x: [word for word in word_tokenize(x) if len(word) > 1])

    return review_df

# Streamlit App
def main():
    st.title("Review Data Analysis")
    st.header("Submit Your Review for Analysis")

    # Input review
    input_review = st.text_area("Enter your review:", height=150)

    if st.button("Submit"):
        if input_review:
            # Proses review yang dimasukkan
            review_df = process_reviews(input_review)
            st.write("Preprocessed Review:")
            st.write(review_df['review'].iloc[0])
            
            # Load expert data (assuming CSV file on GitHub or local)
            expert_df = pd.read_csv('Review Expert.csv', sep=";", usecols=["review", "Summary"]).rename(columns={"Summary": "summary_expert"}).head(125)
            
            # TF-IDF Similarity
            merged_df = tfidf_similarity(review_df, expert_df)
            st.write("Matching Expert Reviews:")
            st.write(merged_df[['review', 'summary_expert']].head())
            
            # TF-IDF Visualization
            if st.checkbox("Show TF-IDF Visualization"):
                plot_tfidf(review_df, document_index=0)
                
            # Word2Vec Graph
            if st.checkbox("Show Word2Vec Graph"):
                selected_text = review_df['tokenized'].iloc[0]
                adj_matrix, features, graph = GraphWord2Vec(selected_text)
                st.write("Word2Vec Graph Visualization:")
                nx.draw(graph, with_labels=True, font_weight='bold', font_color='brown')
                st.pyplot(plt)
        else:
            st.warning("Please enter a review to analyze.")
        
if __name__ == "__main__":
    main()
