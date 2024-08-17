import streamlit as st
import os
import pandas as pd
import sqlite3
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import joblib  # For saving and loading models
import openai
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Paths for the model and database
MODEL_PATH = "xgboost_lotto_model.pkl"
DB_PATH = "lotto_data.db"

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Create table if it doesn't exist
    c.execute('''
        CREATE TABLE IF NOT EXISTS lotto_numbers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            number1 INTEGER,
            number2 INTEGER,
            number3 INTEGER,
            number4 INTEGER,
            number5 INTEGER,
            number6 INTEGER,
            bonus INTEGER,
            date TEXT,
            additional_column TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Add a missing column to the SQLite table if it doesn't exist
def alter_table_add_column():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Add 'additional_column' if it doesn't exist
    try:
        c.execute("ALTER TABLE lotto_numbers ADD COLUMN additional_column TEXT")
    except sqlite3.OperationalError:
        # Column already exists
        pass
    
    conn.commit()
    conn.close()

# Function to save the uploaded CSV file to SQLite database
def save_to_db(df):
    # Drop 'additional_column' if it exists in the DataFrame
    if "additional_column" in df.columns:
        df = df.drop(columns=["additional_column"])
    
    # Convert columns to integers
    df = df.astype({
        "number1": "int",
        "number2": "int",
        "number3": "int",
        "number4": "int",
        "number5": "int",
        "number6": "int",
        "bonus": "int"
    })

    # Format the date column
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    # Save to SQLite database
    conn = sqlite3.connect(DB_PATH)
    df.to_sql('lotto_numbers', conn, if_exists='append', index=False)
    conn.close()

# Function to train the XGBoost model
def train_xgboost_model():
    # Load data from SQLite
    conn = sqlite3.connect(DB_PATH)
    all_data = pd.read_sql('SELECT * FROM lotto_numbers', conn)
    conn.close()

    # Prepare features and target variables
    X = all_data[["number1", "number2", "number3", "number4", "number5", "number6"]].shift(1).dropna()
    y = all_data[["number1", "number2", "number3", "number4", "number5", "number6"]][1:]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the XGBoost model
    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, MODEL_PATH)
    st.success("모델이 성공적으로 학습되고 저장되었습니다.")

    # Evaluate model performance
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    st.write(f"평균 제곱 오차(MSE): {mse:.2f}")

# Function to make predictions using the XGBoost model
def predict_with_xgboost_model():
    # Load the trained model
    model = joblib.load(MODEL_PATH)

    # Load the latest data
    conn = sqlite3.connect(DB_PATH)
    all_data = pd.read_sql('SELECT * FROM lotto_numbers', conn)
    conn.close()

    # Make predictions based on the latest data
    X_latest = all_data[["number1", "number2", "number3", "number4", "number5", "number6"]].iloc[-1].values.reshape(1, -1)
    next_numbers = model.predict(X_latest)
    
    # Ensure predictions are within the valid range (1 to 45)
    next_numbers = [max(1, min(45, int(round(num)))) for num in next_numbers.flatten()]
    
    return next_numbers

# Function to analyze the predictions and generate a report
def analyze_predictions(predictions, y_test, predictions_full):
    # Calculate error metrics
    mse = mean_squared_error(y_test, predictions_full)
    mae = mean_absolute_error(y_test, predictions_full)
    std_dev = np.std(predictions)
    correlation_matrix = np.corrcoef(predictions, y_test)

    # Prepare the analysis report
    report = (
        f"### 예측 번호 통계 분석\n"
        f"- 평균 제곱 오차(MSE): {mse:.2f}\n"
        f"- 평균 절대 오차(MAE): {mae:.2f}\n"
        f"- 표준 편차(Standard Deviation): {std_dev:.2f}\n"
        f"- 상관관계(Correlation Coefficient):\n{correlation_matrix}\n"
    )

    return report

# Function to generate an explanation using the OpenAI API
def generate_explanation(predictions, analysis_report):
    prompt = (
        f"The following numbers were predicted as the next lottery numbers: "
        f"{', '.join(map(str, predictions))}. "
        "The analysis report is as follows:\n"
        f"{analysis_report}\n"
        "Explain in detail why these numbers were predicted based on the model's analysis of past lottery data, "
        "considering patterns, trends, and any relevant statistical analysis."
    )

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant.write in korean"},
            {"role": "user", "content": prompt} 
        ]
    )

    return response['choices'][0]['message']['content'].strip()

# Set up the Streamlit app layout
st.title("로또 당첨번호 예측 시스템")

# Initialize database and update table schema
init_db()
alter_table_add_column()

# File upload section
st.header("이전 당첨 번호 업로드")
uploaded_file = st.file_uploader("이전 당첨 번호가 포함된 CSV 파일을 업로드하세요", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Rename columns to match SQLite table schema
    df.columns = ["number1", "number2", "number3", "number4", "number5", "number6", "bonus", "date", "additional_column"]
    
    st.write(df)
    save_to_db(df)  # Save uploaded file to SQLite database
    st.success("파일이 성공적으로 업로드되고 데이터베이스에 저장되었습니다.")

    # Train the model if not already trained
    if not os.path.exists(MODEL_PATH):
        train_xgboost_model()

# Prediction button
if st.button("다음 당첨 번호 예측"):
    if os.path.exists(MODEL_PATH):
        predictions = predict_with_xgboost_model()  # Make predictions
        st.write(f"예측된 다음 당첨 번호: {', '.join(map(str, predictions))}")
        
        # Generate full predictions for analysis
        conn = sqlite3.connect(DB_PATH)
        all_data = pd.read_sql('SELECT * FROM lotto_numbers', conn)
        conn.close()
        
        X = all_data[["number1", "number2", "number3", "number4", "number5", "number6"]].shift(1).dropna()
        y = all_data[["number1", "number2", "number3", "number4", "number5", "number6"]][1:]
        
        model = joblib.load(MODEL_PATH)
        predictions_full = model.predict(X)
        
        # Generate analysis report
        analysis_report = analyze_predictions(predictions, y, predictions_full)
        
        # Display the analysis report
        st.write("### 예측 번호 통계 분석")
        st.write(analysis_report)
        
        # Generate an explanation using OpenAI API
        explanation = generate_explanation(predictions, analysis_report)
        st.write("### 예측 번호 설명")
        st.write(explanation)
    else:
        st.error("모델이 아직 학습되지 않았습니다. CSV 파일을 업로드하고 모델을 학습하세요.")

# Final app message
st.write("앱이 이제 로또 당첨 번호를 예측하고, 그 이유를 설명할 준비가 되었습니다!")
