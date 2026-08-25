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
