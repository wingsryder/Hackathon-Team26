# Analysis page for the Streamlit app
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/wingsryder/Hackathon-Team26/main/Telcom-Customer-Churn.csv'
    df = pd.read_csv(url)
    return df

def main():
    st.title("Data Analysis")

    # Load and display dataset
    df = load_data()
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # Display basic statistics
    st.write("### Dataset Statistics")
    st.write(df.describe())

    # Visualize churn distribution
    st.write("### Churn Distribution")
    churn_counts = df['Churn'].value_counts()
    fig, ax = plt.subplots()
    churn_counts.plot(kind='bar', ax=ax, color=['skyblue', 'orange'])
    ax.set_title("Churn Distribution")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Count")
    st.pyplot(fig)

if __name__ == "__main__":
    main()
