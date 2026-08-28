import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score
import mlflow, mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
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

# Features toSave to SQLite using sqlite3
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

# -------------------------------
# M3: Experiments with MLflow
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

mlflow.set_experiment("CustomerSupportClassifier")

models = {
    "LogisticRegression": LogisticRegression(max_iter=300, C=1.0),
    "MultinomialNB": MultinomialNB(),
    "LinearSVM": LinearSVC(C=1.0)
}

results = []
best_model, best_f1 = None, 0

for name, clf in models.items():
    with mlflow.start_run(run_name=name):
        clf.fit(X_train_vec, y_train)
        preds = clf.predict(X_test_vec)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="weighted")
        rec = recall_score(y_test, preds, average="weighted")
        f1 = f1_score(y_test, preds, average="weighted")

        mlflow.log_param("model", name)
        mlflow.log_param("max_features", 5000)
        mlflow.log_param("ngram_range", "(1,2)")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(clf, name=f"{name}_Model")

        results.append([name, acc, prec, rec, f1])
        if f1 > best_f1:
            best_f1, best_model = f1, clf

# Comparison table 
results_df = pd.DataFrame(results, columns=["Model","Accuracy","Precision","Recall","F1"])
print("\nModel Comparison Table:\n", results_df)

print("✅ Selected best model:", results_df.loc[results_df["F1"].idxmax(),"Model"])

joblib.dump(best_model, "support_model.pkl")
joblib.dump(vectorizer, "support_vectorizer.pkl")

# -------------------------------
# M4: REST API with FastAPI
# -------------------------------
app = FastAPI(
    title="Customer Support Ticket Classifier API",
    description="Predict the category of a customer support ticket from its subject and description.",
    version="1.0.0",
)

class TicketInput(BaseModel):
    subject: str = Field(..., examples=["Payment problem"])
    description: str = Field(..., examples=["My payment was deducted twice."])

@app.post("/predict_with_logging")
def predict_and_log(input: TicketInput):
    text = input.subject + " " + input.description
    vec = vectorizer.transform([clean_text(text)])
    pred = best_model.predict(vec)[0]
    log_prediction(text, pred)
    return {"prediction": str(pred)}

@app.post("/drift_score")
def drift_score(threshold: float = 0.1):
    logs = pd.read_csv(LOG_FILE, names=["time","text","pred"])
    pred_dist = logs["pred"].value_counts(normalize=True)
    train_dist = pd.Series(y_train).value_counts(normalize=True)
    drift = (pred_dist - train_dist).abs().sum()
    retrain = drift > threshold
    return {"drift_score": float(drift), "trigger_retrain": retrain}

def retrain_model():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    new_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_train_vec = new_vectorizer.fit_transform(X_train)
    best_model.fit(X_train_vec, y_train)
    joblib.dump(best_model, "support_model.pkl")
    joblib.dump(new_vectorizer, "support_vectorizer.pkl")
    print("✅ Model retrained.")
