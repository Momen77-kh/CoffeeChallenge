import os
import pandas as pd
from textblob import TextBlob
from dotenv import load_dotenv

# ------------------------
# 1. Load environment variables
# ------------------------
load_dotenv()

# Get file paths from .env (safer than hardcoding)
INPUT_FILE = os.getenv("INPUT_FILE")
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "results_with_sentiment.csv")

if not INPUT_FILE or not os.path.exists(INPUT_FILE):
    raise FileNotFoundError("Input file not found. Check INPUT_FILE in your .env")

# ------------------------
# 2. Load dataset safely
# ------------------------
df = pd.read_csv(INPUT_FILE)
print(f"Dataset loaded successfully with {len(df)} rows")
print(f"Available columns: {list(df.columns)}")

# ------------------------
# 3. Sentiment Analysis Function
# ------------------------
def get_sentiment(text: str) -> str:
    """
    Analyze sentiment of text safely and return classification
    """
    if pd.isna(text) or not str(text).strip():
        return "Neutral"

    try:
        polarity = TextBlob(str(text)).sentiment.polarity

        # Configurable thresholds
        POS_THRESHOLD = float(os.getenv("POS_THRESHOLD", 0.1))
        NEG_THRESHOLD = float(os.getenv("NEG_THRESHOLD", -0.1))

        if polarity > POS_THRESHOLD:
            return "Positive"
        elif polarity < NEG_THRESHOLD:
            return "Negative"
        else:
            return "Neutral"

    except Exception:
        # Do not expose original text or raw error
        return "Neutral"

# ------------------------
# 4. Process Coffee Notes Columns
# ------------------------
note_columns = [col for col in df.columns if "note" in col.lower() and "coffee" in col.lower()]

if not note_columns:
    print("No columns with 'coffee' and 'note' found!")
else:
    for col in note_columns:
        sentiment_col_name = f"{col.replace(' ', '_').replace('-', '_')}_Sentiment"
        print(f"Processing column: {col}")
        df[sentiment_col_name] = df[col].apply(get_sentiment)

        # Show sentiment distribution
        print(f"Sentiment distribution for {col}:")
        print(df[sentiment_col_name].value_counts())

# ------------------------
# 5. Save results securely
# ------------------------
# Never overwrite the original file
df.to_csv(OUTPUT_FILE, index=False)
print(f"Results saved to: {OUTPUT_FILE}")

# ------------------------
# 6. Summary
# ------------------------
print(f"Final dataset has {len(df)} rows and {len(df.columns)} columns")
sentiment_columns = [col for col in df.columns if "sentiment" in col.lower()]
print(f"Sentiment columns created: {sentiment_columns}")
