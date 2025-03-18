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

# Fungsi untuk mengambil data dari GitHub
def load_data_from_github():
    review_url = 'https://raw.githubusercontent.com/Syawal1997/streamlit-dashboard-GCN-Word2Vec/main/20191002-reviews.xlsx'
    human_review_url = 'https://raw.githubusercontent.com/Syawal1997/streamlit-dashboard-GCN-Word2Vec/main/human_review.xlsx'

    review_df = pd.read_excel(review_url)
    human_review_df = pd.read_excel(human_review_url)
    
    return review_df, human_review_df

# Preprocessing function
def preprocess_text(text):
    factory = StemmerFactory()
    stopword_factory = StopWordRemoverFactory()
    stemmer = factory.create_stemmer()
    stopword_remover = stopword_factory.create_stop_word_remover()
    
    text = re.sub(r'(.)\1+', r'\1', text)  # Remove duplicate characters
    text = stopword_remover.remove(text)   # Remove stopwords
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
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

# Fungsi untuk mengambil data dan memprosesnya
def process_reviews():
    review_df, human_review_df = load_data_from_github()

    # Preprocessing reviews
    review_df = review_df.loc[:, ["itemId", "reviewTitle", "reviewContent"]]
    review_df = review_df.replace(['nan', 'null'], "").fillna("")
    review_df = review_df.replace(r'[^\x00-\x7F]+', '', regex=True)
    review_df = review_df.drop_duplicates()
    review_df["review"] = review_df["reviewTitle"].astype(str) + ' ' + review_df["reviewContent"].astype(str)
    
    review_df["review"] = review_df["review"].apply(preprocess_text)
    
    # Tokenization
    nltk.download('punkt')
    from nltk.tokenize import word_tokenize
    review_df["tokenized"] = review_df["review"].apply(lambda x: [word for word in word_tokenize(x) if len(word) > 1])
    
    # Merge with human review data
    review_df = review_df.groupby(['itemId'])["review"].agg(lambda x: ' '.join(x)).reset_index()
    review_df = review_df.merge(human_review_df, on='itemId')

    return review_df

# Fungsi untuk melakukan perhitungan TF-IDF dan cosine similarity
def tfidf_similarity(review_df):
    vectorizer = TfidfVectorizer()
    tfidf_matrix_reviews = vectorizer.fit_transform(review_df['review'])
    
    df_expert = pd.read_csv('Review Expert.csv', sep=";", usecols=["review", "Summary"]).rename(columns={"Summary": "summary_expert"}).head(125)
    df_expert['processed_review'] = df_expert['review'].apply(lambda x: x.lower())
    
    tfidf_matrix_expert = vectorizer.transform(df_expert['processed_review'])
    cosine_similarities = cosine_similarity(tfidf_matrix_expert, tfidf_matrix_reviews)
    
    matched_indices = [cosine_similarities[i].argmax() for i in range(len(df_expert))]
    merged_df = pd.DataFrame()

    for i in range(len(df_expert)):
        expert_row = df_expert.iloc[i]
        review_row = review_df.iloc[matched_indices[i]]
        combined_row = pd.concat([expert_row, review_row])
        merged_df = pd.concat([merged_df, combined_row.to_frame().T], ignore_index=True)
    
    return merged_df

# Streamlit App
def main():
    st.title("Review Data Analysis")

    st.header("1. Preprocessing and Data Loading")
    review_df = process_reviews()
    st.write(review_df.head())

    st.header("2. TF-IDF Similarity Analysis")
    merged_df = tfidf_similarity(review_df)
    st.write(merged_df.head())

    st.header("3. TF-IDF Visualization")
    if st.checkbox("Show TF-IDF Visualization"):
        document_index = st.slider("Select Document Index", 0, len(review_df) - 1, 0)
        plot_tfidf(review_df, document_index)

    st.header("4. Word2Vec Graph")
    if st.checkbox("Show Word2Vec Graph"):
        selected_itemId = st.selectbox("Select ItemId", review_df['itemId'].unique())
        selected_text = review_df.loc[review_df['itemId'] == selected_itemId, 'tokenized'].values[0]
        adj_matrix, features, graph = GraphWord2Vec(selected_text)
        
        st.write("Graph visualization:")
        nx.draw(graph, with_labels=True, font_weight='bold', font_color='brown')
        st.pyplot(plt)

if __name__ == "__main__":
    main()
