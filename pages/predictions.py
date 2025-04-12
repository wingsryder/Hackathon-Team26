import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
    df.drop('customerID', axis=1, inplace=True)
    df.replace(" ", pd.NA, inplace=True)
    df.dropna(inplace=True)
    df['TotalCharges'] = df['TotalCharges'].astype(float)
    df = pd.get_dummies(df, drop_first=True)

    # Split data
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a simple model
    model = LogisticRegression()
    model.fit(X_train, y_train)

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

if __name__ == "__main__":
    main()
