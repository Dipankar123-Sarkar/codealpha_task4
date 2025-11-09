"""
Simple runner for sentiment and emotion analysis.
Usage:
  python run_analysis.py --input data/sample_amazon.csv --out results

The input CSV should have a `text` column. An optional `date` column helps trend plots.
"""
import argparse
import os
from src.analysis import analyze_file, ensure_nltk_resources


def main():
    parser = argparse.ArgumentParser(description="Run sentiment + emotion analysis on a CSV file")
    parser.add_argument("--input", required=True, help="Path to input CSV (must contain 'text' column)")
    parser.add_argument("--out", default="results", help="Output directory to save results")
    parser.add_argument("--method", choices=["vader","textblob"], default="vader", help="Sentiment method to use")
    args = parser.parse_args()

    ensure_nltk_resources()

    os.makedirs(args.out, exist_ok=True)
    print(f"Analyzing {args.input} using {args.method}...")
    summary_path, details_path, plot_path = analyze_file(args.input, args.out, method=args.method)
    print("Done. Outputs:")
    print(" - summary:", summary_path)
    print(" - detailed results:", details_path)
    print(" - plot:", plot_path)


if __name__ == "__main__":
    main()
