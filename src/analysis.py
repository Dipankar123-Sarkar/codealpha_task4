import os
import re
import csv
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')  # headless-friendly
import matplotlib.pyplot as plt
import seaborn as sns

from nltk import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

from src.lexicons import load_sample_lexicon


def ensure_nltk_resources():
    """Download NLTK resources used by the toolkit if missing."""
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    # newer NLTK installs may require punkt_tab for sentence tokenization
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')
    try:
        nltk.data.find('sentiment/vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon')


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'text' not in df.columns:
        raise ValueError("Input CSV must contain a 'text' column")
    return df


def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def sentiment_vader(texts: List[str]) -> List[str]:
    sia = SentimentIntensityAnalyzer()
    labels = []
    for t in texts:
        s = sia.polarity_scores(str(t))
        comp = s['compound']
        if comp >= 0.05:
            labels.append('positive')
        elif comp <= -0.05:
            labels.append('negative')
        else:
            labels.append('neutral')
    return labels


def sentiment_textblob(texts: List[str]) -> List[str]:
    labels = []
    for t in texts:
        tb = TextBlob(str(t))
        p = tb.sentiment.polarity
        if p > 0.05:
            labels.append('positive')
        elif p < -0.05:
            labels.append('negative')
        else:
            labels.append('neutral')
    return labels


def emotion_from_lexicon(texts: List[str], lexicon: Dict[str, List[str]]) -> List[Dict[str,int]]:
    """Return per-text emotion counts mapping emotion->count using provided lexicon.
    lexicon: mapping word -> list of emotions (e.g. 'happy' -> ['joy'])
    Output: list of dicts for each text.
    """
    results = []
    for t in texts:
        tokens = [w.lower() for w in word_tokenize(str(t))]
        counter = Counter()
        for w in tokens:
            if w in lexicon:
                for emo in lexicon[w]:
                    counter[emo] += 1
        results.append(dict(counter))
    return results


def aggregate_results(df: pd.DataFrame, sentiment_col: str, emotion_col: str) -> Tuple[pd.DataFrame, Dict]:
    # overall sentiment distribution
    dist = df[sentiment_col].value_counts().rename_axis('sentiment').reset_index(name='count')

    # top emotions
    all_emotions = Counter()
    for emo_map in df[emotion_col].fillna({}):
        if isinstance(emo_map, dict):
            all_emotions.update(emo_map)
    top_emotions = pd.DataFrame(all_emotions.most_common(), columns=['emotion','count'])

    summary = {'sentiment_distribution': dist, 'top_emotions': top_emotions}
    return dist, summary


def plot_trends(df: pd.DataFrame, out_path: str, sentiment_col: str = 'sentiment') -> str:
    # If date column exists, try to plot trends by day
    if 'date' in df.columns:
        try:
            tmp = df.copy()
            tmp['date'] = pd.to_datetime(tmp['date'])
            tmp['day'] = tmp['date'].dt.date
            trend = tmp.groupby(['day', sentiment_col]).size().reset_index(name='count')
            plt.figure(figsize=(10,4))
            sns.lineplot(data=trend, x='day', y='count', hue=sentiment_col, marker='o')
            plt.title('Sentiment trends over time')
            plt.tight_layout()
            out_file = os.path.join(out_path, 'sentiment_trends.png')
            plt.savefig(out_file)
            plt.close()
            return out_file
        except Exception:
            pass
    # Otherwise create a simple bar chart of sentiment distribution
    dist = df[sentiment_col].value_counts()
    plt.figure(figsize=(6,4))
    sns.barplot(x=dist.index, y=dist.values)
    plt.title('Sentiment distribution')
    plt.tight_layout()
    out_file = os.path.join(out_path, 'sentiment_trends.png')
    plt.savefig(out_file)
    plt.close()
    return out_file


def analyze_file(input_csv: str, out_dir: str, method: str = 'vader') -> Tuple[str,str,str]:
    df = load_csv(input_csv)
    df['text'] = df['text'].fillna('').astype(str).apply(preprocess_text)

    if method == 'vader':
        df['sentiment'] = sentiment_vader(df['text'].tolist())
    else:
        df['sentiment'] = sentiment_textblob(df['text'].tolist())

    lex = load_sample_lexicon()
    df['emotions'] = emotion_from_lexicon(df['text'].tolist(), lex)

    # save detailed results
    os.makedirs(out_dir, exist_ok=True)
    detailed_path = os.path.join(out_dir, 'detailed_results.csv')
    # convert emotions dict to string for CSV
    df_out = df.copy()
    df_out['emotions'] = df_out['emotions'].apply(lambda d: ",".join([f"{k}:{v}" for k,v in (d or {}).items()]) )
    df_out.to_csv(detailed_path, index=False)

    # summary
    dist, summary = aggregate_results(df, 'sentiment', 'emotions')
    summary_path = os.path.join(out_dir, 'summary.csv')
    # write a compact summary CSV
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['metric','key','value'])
        for _, row in dist.iterrows():
            writer.writerow(['sentiment_count', row['sentiment'], row['count']])
        for emo, cnt in summary['top_emotions'].itertuples(index=False):
            writer.writerow(['emotion_count', emo, cnt])

    plot_path = plot_trends(df, out_dir, sentiment_col='sentiment')
    return summary_path, detailed_path, plot_path
