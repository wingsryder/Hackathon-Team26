#update
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from joblib import load

# Load dataset
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/wingsryder/Hackathon-Team26/main/Telcom-Customer-Churn.csv'
    df = pd.read_csv(url)
    return df

def main():
    st.title("Predictions")

    # Load and preprocess dataset
    df = load_data()

    # Debugging: Print column names after each preprocessing step
    st.write("### Debug: Column Names After Loading")
    st.write(df.columns.tolist())

    # Drop 'customerID' column
    df.drop('customerID', axis=1, inplace=True)
    st.write("### Debug: Column Names After Dropping 'customerID'")
    st.write(df.columns.tolist())

    # Replace spaces with NaN and drop missing values
    df.replace(" ", pd.NA, inplace=True)
    df.dropna(inplace=True)
    st.write("### Debug: Column Names After Cleaning")
    st.write(df.columns.tolist())

    # Convert 'TotalCharges' to float
    df['TotalCharges'] = df['TotalCharges'].astype(float)
    st.write("### Debug: Column Names After Converting 'TotalCharges'")
    st.write(df.columns.tolist())

    # One-hot encode categorical columns
    df = pd.get_dummies(df, drop_first=True)
    st.write("### Debug: Column Names After One-Hot Encoding")
    st.write(df.columns.tolist())

    # Encode 'Churn' column to numeric values
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Split data
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load pre-trained model
    model_path = f"{selected_model.replace(' ', '_')}_model.joblib"
    try:
        model = load(model_path)
        st.success(f"Loaded pre-trained model: {selected_model}")
    except FileNotFoundError:
        st.error(f"Model file not found: {model_path}. Please ensure the model is saved correctly.")

    # Evaluate model
    y_pred = model.predict(X_test)
    st.write("### Model Performance")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    st.write(f"Precision: {precision_score(y_test, y_pred):.2f}")
    st.write(f"Recall: {recall_score(y_test, y_pred):.2f}")
    st.write(f"F1 Score: {f1_score(y_test, y_pred):.2f}")

    # Make predictions
    st.write("### Make a Prediction")
    user_input = {col: st.text_input(col, "0") for col in X.columns}
    if st.button("Predict"):
        input_df = pd.DataFrame([user_input])
        prediction = model.predict(input_df)[0]
        st.write(f"Prediction: {'Churn' if prediction == 1 else 'No Churn'}")

    # Debugging: Print column names to verify 'Churn' column
    st.write("### Debug: Column Names")
    st.write(df.columns.tolist())

if __name__ == "__main__":
    main()
