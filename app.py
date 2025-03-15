import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from gensim.models import Word2Vec
import re
import string
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from torch_geometric.nn import GCNConv
import torch.optim as optim
import torch.nn.functional as F

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords

# Initialize Streamlit app
st.title("GCN Model for Text Review Analysis")

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
    return text

# Function to remove punctuation
def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

# Load stopwords from NLTK
stop_words = set(stopwords.words('indonesian'))

# Function to remove stopwords
def stopword_removal(text):
    words = text.split()
    words_filtered = [word for word in words if word not in stop_words]
    return ' '.join(words_filtered)

# Tokenization
def tokenized_text(text):
    return nltk.word_tokenize(text)

# Preprocessing pipeline
def preprocess_review(text):
    text = preprocess_text(text)
    text = remove_punctuation(text)
    text = stopword_removal(text)
    return tokenized_text(text)

# Graph creation based on Word2Vec model
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

# Adjust features to match the number of nodes in adjacency matrix
def AdjustFeatures(f, a):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = a.astype(np.float32)
    f = torch.FloatTensor(f).to(device)

    if f.shape[0] > a.shape[0]:
        f = f[:a.shape[0], :]
    elif f.shape[0] < a.shape[0]:
        a = a[:f.shape[0], :f.shape[0]]

    return f, a, device

# Define a simple GCN model
class GCN(nn.Module):
    def __init__(self, in_channels, out_channels):  
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, out_channels)
        self.conv2 = GCNConv(out_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# Streamlit UI to enter a review
review_text = st.text_area("Enter your review:")

# Button to submit the review
if st.button('Submit Review'):

    if review_text:
        # Preprocess the review
        tokenized_review = preprocess_review(review_text)
        
        # Create graph using Word2Vec
        adj_matrix, features, graph = GraphWord2Vec(tokenized_review)
        
        # Adjust the features and adjacency matrix
        features, adj_matrix, device = AdjustFeatures(features, adj_matrix)

        # Create the graph and plot
        plt.figure(figsize=(8, 6))
        nx.draw(graph, with_labels=True, font_weight='bold', font_color='brown')
        st.pyplot(plt)

        # Define GCN model (for demo, we're using a simple one with 2 layers)
        model = GCN(in_channels=features.shape[1], out_channels=64).to(device)

        # Placeholder edge_index (you can compute this based on the graph)
        edge_index = torch.tensor(np.array(np.nonzero(adj_matrix)), dtype=torch.long).to(device)

        # Perform forward pass through GCN
        output = model(features, edge_index)

        st.write("Output after GCN processing:", output)

    else:
        st.warning("Please enter a review.")
