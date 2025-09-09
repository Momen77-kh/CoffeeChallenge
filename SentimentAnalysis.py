#!/usr/bin/env python3
"""
CSV Sentiment Analyzer
Reads INPUT_FILE from .env, analyzes 'Coffee D - Notes' column,
and updates the same CSV file with a new sentiment column
"""

import pandas as pd
import warnings
import os
from transformers import pipeline
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# Load .env variables
load_dotenv()
INPUT_FILE = os.getenv("INPUT_FILE", "").strip()

class SentimentAnalyzer:
    """Simple sentiment analyzer for user notes"""

    def __init__(self):
        print("Loading sentiment analysis model...")
        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=-1,
                truncation=True,
                max_length=512
            )
            print("Model loaded successfully!")
        except Exception:
            print("Using fallback model...")
            self.sentiment_pipeline = pipeline("sentiment-analysis")
            print("Fallback model loaded!")

    def analyze_text(self, text):
        """Analyze sentiment of a single text and return 'Positive', 'Negative', or 'Neutral'"""
        if pd.isna(text) or not str(text).strip():
            return "Neutral"
        try:
            clean_text = str(text).strip()[:500]
            result = self.sentiment_pipeline(clean_text)[0]
            label = result["label"].upper()
            if label in ["POSITIVE", "POS"]:
                return "Positive"
            elif label in ["NEGATIVE", "NEG"]:
                return "Negative"
            else:
                return "Neutral"
        except Exception:
            return "Neutral"

    def analyze_csv_inplace(self, input_file):
        """Analyze 'Coffee D - Notes' column in the CSV and save changes to the same file"""
        if not os.path.exists(input_file):
            print(f"File not found: {input_file}")
            return None

        df = pd.read_csv(input_file)

        target_col = "Coffee D - Notes"
        if target_col not in df.columns:
            print(f"Column '{target_col}' not found in CSV!")
            print(f"Available columns: {list(df.columns)}")
            return df

        # Analyze the column and add a new sentiment column
        df[f"{target_col}_Sentiment"] = [self.analyze_text(text) for text in df[target_col]]

        # Save changes to the same file
        df.to_csv(input_file, index=False)
        print(f"File updated successfully with new column: '{target_col}_Sentiment'")
        return df



