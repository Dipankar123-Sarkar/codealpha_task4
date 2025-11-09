# Sentiment & Emotion Analysis Toolkit

This small project demonstrates sentiment classification (positive/negative/neutral) and lexicon-based emotion detection for text data from sources like Amazon reviews, social media, and news.

What's included
- `src/analysis.py` — core functions: data loading, preprocessing, VADER/TextBlob sentiment, lexicon-based emotion detection, aggregation and plotting.
- `src/lexicons.py` — small sample emotion lexicon (NRC-like subset) and loader hook.
- `run_analysis.py` — CLI to run analysis on CSV files with a `text` column (and optional `date`).
- `data/` — small sample CSVs for Amazon reviews, social posts, and news headlines.
- `tests/` — pytest tests for basic functionality.

Quick start (Windows PowerShell):

1. Create a virtual environment and activate it:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the sample analysis:

```powershell
python run_analysis.py --input data/sample_amazon.csv --out results
```

Outputs: `results/summary.csv`, `results/sentiment_trends.png`, and `results/detailed_results.csv`.

Notes & next steps
- This is a lightweight starter. For production, add larger lexicons (NRC), use transformer models (Hugging Face), and scale with batching.
- The runner supports any CSV with a `text` column; if `date` exists it will produce time-series trends.
