import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from gensim.models import Word2Vec
import re
import string
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import RegexpTokenizer, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.nn import GCNConv
import torch.optim as optim
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

# Download necessary NLTK data files
nltk.download('stopwords')
nltk.download('punkt')

# Load pre-trained models
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Streamlit Title
st.title("GCN Model with Text Summarization and Analysis")

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    return text

# Load stopwords from NLTK
stop_words = set(stopwords.words('indonesian'))

def stopword_removal(text):
    words = text.split()
    words_filtered = [word for word in words if word not in stop_words]
    return ' '.join(words_filtered)

# Tokenizer (use regex to avoid punkt error)
tokenizer = RegexpTokenizer(r'\w+')

def tokenized_text(text):
    return tokenizer.tokenize(text)

# Preprocessing pipeline
def preprocess_review(text):
    text = preprocess_text(text)
    text = stopword_removal(text)
    return tokenized_text(text)

# Text Summarization Function using TextRank
def summarize_text(text, num_sentences=3):
    try:
        sentences = sent_tokenize(text)
    except LookupError:
        st.warning("Tokenizer Punkt tidak ditemukan, menggunakan metode alternatif.")
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    
    if len(sentences) <= num_sentences:
        return text  # Return original text if too short
    
    vectorizer = TfidfVectorizer()
    sentence_vectors = vectorizer.fit_transform(sentences)
    
    sim_matrix = cosine_similarity(sentence_vectors)
    
    graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(graph)
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    
    summary = ' '.join([s[1] for s in ranked_sentences[:num_sentences]])
    return summary

# TF-IDF Analysis
def tfidf_analysis(text):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray()[0]
    return ' '.join(sorted(feature_names, key=lambda x: scores[vectorizer.vocabulary_[x]], reverse=True)[:5])

# Word2Vec Analysis
def word2vec_analysis(text):
    words = tokenized_text(text)
    w2v_model = Word2Vec([words], vector_size=64, window=2, min_count=1, sg=0)
    word_vectors = {word: w2v_model.wv[word].tolist() for word in words if word in w2v_model.wv}
    return ' '.join(list(word_vectors.keys())[:5])

# Sentence Embeddings using SentenceTransformer (like GloVe or LLMs)
def sentence_embedding(text):
    embedding = sentence_model.encode(text)
    return embedding.tolist()

# Streamlit UI
txt_input = st.text_area("Enter your text for summarization and analysis:")

if st.button('Process Text'):
    if txt_input:
        # Summarization
        summary = summarize_text(txt_input)
        st.subheader("Summarized Text (TextRank):")
        st.write(summary)
        
        # TF-IDF Analysis
        tfidf_result = tfidf_analysis(summary)
        st.subheader("TF-IDF Based Summary:")
        st.write(tfidf_result)
        
        # Word2Vec Analysis
        word2vec_result = word2vec_analysis(summary)
        st.subheader("Word2Vec Based Summary:")
        st.write(word2vec_result)
        
        # Sentence Embeddings
        embedding_result = sentence_embedding(summary)
        st.subheader("Sentence Embedding (LLM/GloVe-based) Summary:")
        st.write(embedding_result[:5])
    else:
        st.warning("Please enter some text.")
