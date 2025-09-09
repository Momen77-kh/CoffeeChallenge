from SentimentAnalysis import SentimentAnalyzer
from dotenv import load_dotenv
import os

# Load .env variables
load_dotenv()
INPUT_FILE = os.getenv("INPUT_FILE", "").strip()

if not INPUT_FILE:
    print("Please set INPUT_FILE in your .env")
else:
    analyzer = SentimentAnalyzer()
    analyzer.analyze_csv_inplace(INPUT_FILE)


