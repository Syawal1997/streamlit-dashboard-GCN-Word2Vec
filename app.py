import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from gensim.models import Word2Vec
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import re
import string
import matplotlib.pyplot as plt

# Function to clean and preprocess the review text
def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove stopwords
    factory = StopWordRemoverFactory()
    stopword = factory.create_stop_word_remover()
    text_clean = stopword.remove(text)
    
    # Apply stemming
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    text = stemmer.stem(text_clean)
    
    return text

# Function to generate Graph using Word2Vec
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

# Streamlit UI
st.title("Review Processing with GCN")

# Input from user
review_input = st.text_area("Masukkan Review Anda:", height=200)

if st.button("Proses Review"):
    if review_input:
        # Preprocess review text
        processed_review = preprocess_text(review_input)
        tokenized_review = processed_review.split()

        # Create Graph with Word2Vec
        adj_matrix, features, graph = GraphWord2Vec(tokenized_review)

        # Adjust Features and Adjacency Matrix to match GCN model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        features_tensor = torch.FloatTensor(features).to(device)
        adj_matrix_tensor = torch.FloatTensor(adj_matrix).to(device)

        # Display Graph Visualization
        nx.draw(graph, with_labels=True, font_weight='bold', font_color='brown')
        st.pyplot(plt.gcf())

        # Display adjacency matrix (optional)
        st.write("Adjacency Matrix:")
        st.write(adj_matrix)

        # Example output (dummy result, replace with your GCN model's output)
        st.subheader("Hasil Ringkasan dengan Model GCN:")
        # For example, we can just output a dummy text for now
        st.write("Ringkasan: Review Anda telah diproses melalui GCN!")
    else:
        st.warning("Masukkan review terlebih dahulu.")
