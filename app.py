import streamlit as st
import torch
import torch.nn as nn
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from collections import defaultdict
from gensim.models import Word2Vec
import string
from nltk.tokenize import word_tokenize
import numpy as np
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.metrics.pairwise import cosine_similarity

# Function to preprocess text (you can adjust this based on your needs)
def preprocess_text(text):
    text = text.lower()
    return text

# Remove duplicate characters
def remove_duplicate_chars(text):
    special_words = ['good']
    for i, word in enumerate(special_words):
        text = text.replace(word, f"_{i}_")
    no_double_text = re.sub(r'(.)\1+', r'\1', text)
    for i, word in enumerate(special_words):
        no_double_text = no_double_text.replace(f"_{i}_", word)
    return no_double_text

# Tokenization and stopword removal
def stopword_removal(text):
    factory = StopWordRemoverFactory()
    stopword = factory.create_stop_word_remover()
    text_clean = stopword.remove(text)
    return text_clean

def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def tokenized_text(text):
    tokenized_text = word_tokenize(text)
    tokenized_text = [word for word in tokenized_text if len(word) > 1]
    return tokenized_text

# Graph Creation for Word2Vec and GCN model
def GraphWord2Vec(text):
    # Train Word2Vec model
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

def AdjustFeatures(f, a):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = a.astype(np.float32)
    f = torch.FloatTensor(f).to(device)

    if f.shape[0] > a.shape[0]:
        f = f[:a.shape[0], :]
    elif f.shape[0] < a.shape[0]:
        a = a[:f.shape[0], :f.shape[0]]

    return f, a, device

# Streamlit Application
st.title("GCN Review Analyzer")

# Text input for review
user_review = st.text_area("Enter your review", "Type your review here...")

# Preprocess the review
if user_review:
    # Process the input review
    processed_review = preprocess_text(user_review)
    processed_review = remove_duplicate_chars(processed_review)
    processed_review = stopword_removal(processed_review)
    processed_review = remove_punctuation(processed_review)
    tokenized_review = tokenized_text(processed_review)
    
    # Display the original review
    st.subheader("Original Review")
    st.write(user_review)

    # Display the processed review
    st.subheader("Processed Review")
    st.write(processed_review)

    # Display Tokenized Text
    st.subheader("Tokenized Review")
    st.write(tokenized_review)

    # Graph Generation with Word2Vec
    adj_matrix, features, graph = GraphWord2Vec(tokenized_review)

    # Adjust features and adjacency matrix for the model
    f, a, device = AdjustFeatures(features, adj_matrix)

    # Plot the Word2Vec Graph using NetworkX
    plt.figure(figsize=(10, 8))
    nx.draw(graph, with_labels=True, font_weight='bold', font_color='brown', node_size=500, node_color="skyblue")
    st.subheader("Word2Vec Graph Visualization")
    st.pyplot()

    # TF-IDF Vectorizer and Graph Plotting
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform([processed_review])
    feature_names = tfidf.get_feature_names_out()

    dense_matrix = tfidf_matrix.toarray()
    tfidf_df = pd.DataFrame(dense_matrix, columns=feature_names)

    # Plotting TF-IDF scores
    document_index = 0
    tfidf_scores = tfidf_df.iloc[document_index]
    sorted_tfidf_scores = tfidf_scores.sort_values(ascending=False)
    top_n = 10
    top_tfidf_scores = sorted_tfidf_scores[:top_n]

    plt.figure(figsize=(10, 6))
    plt.bar(top_tfidf_scores.index, top_tfidf_scores.values)
    plt.xlabel("Words")
    plt.ylabel("TF-IDF Score")
    plt.title(f"Top {top_n} Words with Highest TF-IDF Scores")
    plt.xticks(rotation=45, ha='right')
    st.subheader("TF-IDF Score Visualization")
    st.pyplot()

# Add instructions or information to the Streamlit app
st.write(
    """
    **How it works**: 
    - Enter a review in the input box.
    - The app will preprocess the review by removing stopwords, punctuation, and tokenizing it.
    - It will then display the processed review, tokenized words, and visualize the Word2Vec graph and TF-IDF scores.
    """
)
