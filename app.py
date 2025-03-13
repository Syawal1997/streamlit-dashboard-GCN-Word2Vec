import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import networkx as nx
import matplotlib.pyplot as plt
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from nltk.tokenize import word_tokenize
from collections import defaultdict
import requests

# Load datasets from GitHub URL
BASE_PATH = "https://raw.githubusercontent.com/Syawal1997/streamlit-dashboard-GCN-Word2Vec/main/"
DATASET_PATH = "20191002-reviews.xlsx"
HR_PATH = "human_review.xlsx"

def load_data():
    review_df = pd.read_excel(BASE_PATH + DATASET_PATH)
    human_review_df = pd.read_excel(BASE_PATH + HR_PATH)
    return review_df, human_review_df

def preprocess_text(text):
    text = text.lower()
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    return text

def stopword_removal(text):
    factory = StopWordRemoverFactory()
    stopword = factory.create_stop_word_remover()
    return stopword.remove(text)

def tokenized_text(text):
    return [word for word in word_tokenize(text) if len(word) > 1]

def generate_tfidf_summary(reviews):
    tfidf_vectorizer = TfidfVectorizer(max_features=10)
    tfidf_matrix = tfidf_vectorizer.fit_transform(reviews)
    feature_names = np.array(tfidf_vectorizer.get_feature_names_out())
    dense = tfidf_matrix.todense()
    tfidf_scores = dense.tolist()
    return list(zip(feature_names, tfidf_scores[0]))

def generate_word2vec_summary(reviews):
    sentences = [review.split() for review in reviews]
    model = Word2Vec(sentences, min_count=1, vector_size=100, workers=4)
    most_common_words = sorted(model.wv.key_to_index, key=lambda x: model.wv.get_vecattr(x, "count"), reverse=True)
    return most_common_words[:10]

def generate_gcn_summary(reviews):
    G = nx.Graph()
    for review in reviews:
        words = review.split()
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                if not G.has_edge(words[i], words[j]):
                    G.add_edge(words[i], words[j])
    degree_dict = dict(G.degree())
    sorted_degree = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)
    return [item[0] for item in sorted_degree[:10]]

def main():
    st.title("Review Summary Generator")
    
    # Upload file
    st.sidebar.title("Upload Your Data")
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=["xlsx", "csv"])
    
    # Select Model
    model = st.selectbox("Select Summary Model", ["TF-IDF", "Word2Vec", "GCN"])

    # Input Review
    review_input = st.text_area("Masukkan Review", "Tulis review Anda di sini...")
    submit_button = st.button("Generate Summary")

    if submit_button:
        if uploaded_file is not None:
            if uploaded_file.name.endswith('xlsx'):
                data = pd.read_excel(uploaded_file)
            else:
                data = pd.read_csv(uploaded_file)
            
            st.write(f"Loaded {uploaded_file.name} with {len(data)} records.")
        
        # Preprocess input
        review_input_clean = preprocess_text(review_input)
        review_input_clean = stopword_removal(review_input_clean)
        review_tokenized = tokenized_text(review_input_clean)
        
        reviews = [" ".join(review_tokenized)]  # Wrap single review in list for processing

        if model == "TF-IDF":
            summary = generate_tfidf_summary(reviews)
            st.subheader("TF-IDF Summary:")
            for word, score in summary:
                st.write(f"{word}: {score}")

        elif model == "Word2Vec":
            summary = generate_word2vec_summary(reviews)
            st.subheader("Word2Vec Summary:")
            st.write(", ".join(summary))

        elif model == "GCN":
            summary = generate_gcn_summary(reviews)
            st.subheader("GCN Summary:")
            st.write(", ".join(summary))

if __name__ == "__main__":
    main()
