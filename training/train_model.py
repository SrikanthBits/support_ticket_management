import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score
import mlflow, mlflow.sklearn
from fastapi import FastAPI, HTTPException
import joblib, re , string
import os
import json
import sqlite3

# Load dataset
print("SCRIPT STARTED") # <-- Add this line
print("Does file exist?:", os.path.exists('data/customer_support_tickets.csv'))

# -------------------------------
# M2: Data Engineering & Validation
# -------------------------------

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip() 

df = pd.read_csv('data/customer_support_tickets.csv')
print("Dataset loaded successfully.", df.head())

# Features stored in SQLite database

db_path = os.path.abspath("data/customer_support_tickets_features.db")

# Feature stoSave to SQLite using sqlite3
conn = sqlite3.connect(db_path)
df.to_sql("customer_support_tickets", conn, if_exists="replace", index=False)
conn.close()

# engine = create_engine("sqlite:///data/customer_support_tickets_features.db")
# df.to_sql("customer_support_tickets", engine, if_exists="replace", index=False)

print("✅ Table created from CSV headers")


connection = sqlite3.connect('data/customer_support_tickets_features.db')
print(connection.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall())
features = pd.read_sql_query("SELECT * FROM customer_support_tickets", connection)
connection.close()

print("Rows in SQLite database:", len(features))
print("Columns in SQLite database:", features.columns)
print("Data loaded from SQLite database successfully.", features.head())

# Schema validation
expected_cols = [
    "Ticket ID","Customer Name","Customer Email","Customer Age","Customer Gender",
    "Product Purchased","Date of Purchase","Ticket Type","Ticket Subject",
    "Ticket Description","Ticket Status","Resolution","Ticket Priority",
    "Ticket Channel","First Response Time","Time to Resolution","Customer Satisfaction Rating"
]
missing_cols = [c for c in expected_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns: {missing_cols}")

# Duplicate removal
df = df.drop_duplicates(subset=["Ticket ID"])

# Combine text features
df["text"] = df["Ticket Subject"].fillna("") + " " + df["Ticket Description"].fillna("")
df["clean_text"] = df["text"].apply(clean_text)

# Drop invalid/missing target
df = df.dropna(subset=["Ticket Type"])
X = df["clean_text"]
y = df["Ticket Type"]

# Dataset statistics
print("Dataset size:", len(df))
print("Class distribution:\n", y.value_counts())