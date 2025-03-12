import streamlit as st
import pandas as pd

# Load summary results
def load_data():
    return pd.read_csv("summary_results.csv")  # Pastikan file ini sudah ada dari proses sebelumnya

# Streamlit UI
st.title("E-commerce Review Summarization Dashboard")

# Load Data
summary_results = load_data()

# Select Product
product_ids = summary_results['itemId'].unique()
selected_product = st.selectbox("Select Product ID", product_ids)

# Filter Data
filtered_data = summary_results[summary_results['itemId'] == selected_product]

st.subheader("Summarization Results")
st.write(f"**Product ID:** {selected_product}")

# Display Summaries
if not filtered_data.empty:
    st.write("### Word2Vec Summary")
    st.write(filtered_data['Word2Vec'].values[0])
    
    st.write("### TFIDF Summary")
    st.write(filtered_data['TFIDF'].values[0])
    
    st.write("### Glove Summary")
    st.write(filtered_data['Glove'].values[0])
else:
    st.write("No summary available for this product.")
