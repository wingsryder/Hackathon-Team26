import streamlit as st
import pandas as pd
from utils.data_loader import load_data
from utils.model_loader import evaluate_model, get_test_predictions
from visualizations.plot_functions import plot_metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

def load_css():
    with open("static/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def load_js():
    with open("static/script.js") as f:
        st.markdown(f"<script>{f.read()}</script>", unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="ML Dashboard", layout="wide")
    load_css()
    load_js()

    # Sidebar for filters or model selection
    st.sidebar.title("Options")
    selected_model = st.sidebar.selectbox("Select Model", ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting", "SVM"])

    # Main sections
    st.title("ML Dashboard")

    # Dataset display
    st.header("Dataset")
    df = load_data()
    st.dataframe(df.head())

    # Model evaluation metrics
    st.header("Model Evaluation")
    metrics = evaluate_model(selected_model)

    # Tabs for better organization
    tab1, tab2, tab3 = st.tabs(["Metrics Table", "Bar Plot", "Confusion Matrix"])

    # Load results from the notebook
    @st.cache_data
    def load_results():
        # Simulate loading results from the notebook
        results = {
            'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting', 'SVM'],
            'Accuracy': [0.85, 0.83, 0.87, 0.88, 0.84],
            'Precision': [0.82, 0.80, 0.86, 0.87, 0.81],
            'Recall': [0.78, 0.76, 0.84, 0.85, 0.79],
            'F1 Score': [0.80, 0.78, 0.85, 0.86, 0.80]
        }
        return pd.DataFrame(results)

    # Display results in the app
    st.header("Model Results from Notebook")
    results_df = load_results()
    st.dataframe(results_df)

    # Update Metrics Table to load data from notebook
    with tab1:
        st.write("### Metrics Table")
        st.table(results_df)  # Use notebook results

    # Update Bar Plot to use notebook results
    with tab2:
        st.write("### F1 Score Comparison")
        fig, ax = plt.subplots()
        sns.barplot(data=results_df, x='Model', y='F1 Score', ax=ax, palette="viridis")
        ax.set_title("F1 Score by Model")
        ax.set_ylabel("F1 Score")
        ax.set_xlabel("Model")
        st.pyplot(fig)

    # Update Confusion Matrix to use notebook results
    with tab3:
        st.write("### Confusion Matrix")
        # Simulate confusion matrix data
        cm = [[50, 10], [5, 35]]  # Example data
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    # Optional: ROC Curve
    if st.sidebar.checkbox("Show ROC Curve"):
        st.write("### ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_title("Receiver Operating Characteristic")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
        st.pyplot(fig)

if __name__ == "__main__":
    main()