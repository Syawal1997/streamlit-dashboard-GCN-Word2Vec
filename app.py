import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from wordcloud import WordCloud
from rouge_score import rouge_scorer
import sacrebleu

# Assuming merged_df is already loaded
# merged_df = pd.read_csv('your_data.csv')  # Replace with your actual data loading step

# Function to generate WordClouds
def plot_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.tight_layout()
    st.pyplot()

# Function to create a combined histogram plot
def plot_combined_histogram(df, column_names, title, xlabel):
    plt.figure(figsize=(12, 6))
    for col in column_names:
        plt.hist(df[col], bins=20, alpha=0.5, label=col)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    st.pyplot()

# Function to generate scatter plot
def plot_scatter(df, columns, xlabel, ylabel, title):
    plt.figure(figsize=(12, 8))
    for col in columns:
        plt.scatter(df[columns[0]], df[col], label=col, alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    st.pyplot()

# Function to calculate BLEU score
def calculate_bleu(references, hypotheses):
    return sacrebleu.corpus_bleu(hypotheses, references).score

# Function to calculate ROUGE scores
def calculate_rouge_score(ref, hyp):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    return scorer.score(ref, hyp)

# Streamlit UI elements
st.title("Summary Model Evaluation Dashboard")

# Data Summary Section
st.header("Summary Statistics")
st.write("This section shows basic statistics about the character and word counts for each summary model.")

# Assuming merged_df contains 'human_review', 'Word2Vec', 'TFIDF', 'Glove', etc.
text_human_review = ' '.join(merged_df['human_review'].astype(str))
plot_wordcloud(text_human_review, 'Wordcloud of Human Reviews')

# Column-wise word cloud generation for summaries
columns_to_summarize = ['Word2Vec', 'TFIDF', 'Glove', 'Word2Vec By Claude AI', 'TFIDF By Claude AI', 'Glove By Claude AI']
for col in columns_to_summarize:
    text_summary = ' '.join(merged_df[col].astype(str))
    plot_wordcloud(text_summary, f'Wordcloud of {col} Summaries')

# Character distribution histogram
st.header("Character Count Distribution")
char_columns = ['human_review_char_count', 'summary_char_count_Word2Vec', 'summary_char_count_TFIDF',
                'summary_char_count_Glove', 'summary_char_count_Word2Vec_By_Claude_AI',
                'summary_char_count_TFIDF_By_Claude_AI', 'summary_char_count_Glove_By_Claude_AI']
plot_combined_histogram(merged_df, char_columns, 'Character Distribution Across Summaries', 'Character Count')

# Word distribution histogram
st.header("Word Count Distribution")
word_columns = ['human_review_word_count', 'summary_word_count_Word2Vec', 'summary_word_count_TFIDF',
                'summary_word_count_Glove', 'summary_word_count_Word2Vec_By_Claude_AI',
                'summary_word_count_TFIDF_By_Claude_AI', 'summary_word_count_Glove_By_Claude_AI']
plot_combined_histogram(merged_df, word_columns, 'Word Distribution Across Summaries', 'Word Count')

# Scatter plot for character count comparison
st.header("Length Comparison: Human Review vs. Summaries")
columns_for_scatter = ['human_review_char_count', 'summary_char_count_Word2Vec', 'summary_char_count_TFIDF', 
                       'summary_char_count_Glove', 'summary_char_count_Word2Vec_By_Claude_AI', 
                       'summary_char_count_TFIDF_By_Claude_AI', 'summary_char_count_Glove_By_Claude_AI']
plot_scatter(merged_df, columns_for_scatter, 'Human Review Character Count', 'Summary Character Count', 
             'Length Comparison: Human Review vs. Summaries')

# BLEU score evaluation
st.header("BLEU Scores for Summary Models")
bleu_scores = {}
for col in columns_to_summarize:
    references = merged_df['human_review'].astype(str).tolist()
    hypotheses = merged_df[col].astype(str).tolist()
    bleu_scores[col] = calculate_bleu([references], hypotheses)

bleu_df = pd.DataFrame({'Model': list(bleu_scores.keys()), 'BLEU Score': list(bleu_scores.values())})
st.write(bleu_df)

# ROUGE score evaluation
st.header("ROUGE Scores for Summary Models")
rouge_scores = {}
for col in columns_to_summarize:
    rouge_scores[col] = []
    references = merged_df['human_review'].astype(str).tolist()
    hypotheses = merged_df[col].astype(str).tolist()

    for ref, hyp in zip(references, hypotheses):
        rouge_scores[col].append(calculate_rouge_score(ref, hyp)['rouge1'].fmeasure)

rouge_df = pd.DataFrame(rouge_scores)
st.write(rouge_df)

# Entity Relationship Visualization
st.header("Entity Relationship Diagram")
graph = nx.Graph()
for item_id in merged_df['itemId'].unique():
    graph.add_node(item_id, type='item')

for index, row in merged_df.iterrows():
    item_id = row['itemId']
    human_review = row['human_review']
    graph.add_node(f"{item_id}_human_review", type='human_review', review=human_review)
    graph.add_edge(item_id, f"{item_id}_human_review", label="has_review")

    for col in columns_to_summarize:
        summary = row[col]
        graph.add_node(f"{item_id}_{col}_summary", type='summary', summary=summary)
        graph.add_edge(f"{item_id}_human_review", f"{item_id}_{col}_summary", label=f"generated_{col}_summary")

plt.figure(figsize=(12, 8))
pos = nx.spring_layout(graph)
nx.draw(graph, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=8)
edge_labels = nx.get_edge_attributes(graph, 'label')
nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=6)
st.pyplot()
