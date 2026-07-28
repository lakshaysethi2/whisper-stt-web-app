"""Compute word accuracy between reference and hypothesis text files.

Usage: python3 _compute_wer.py <reference.txt> <hypothesis.txt>

Outputs a single line: accuracy_pct wer ref_word_count hyp_word_count
"""
import re
import sys


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def word_wer(ref_words: list[str], hyp_words: list[str]) -> float:
    """Word-level WER using Levenshtein distance on word sequences."""
    m, n = len(ref_words), len(hyp_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )
    return dp[m][n] / m if m > 0 else 1.0


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <reference.txt> <hypothesis.txt>", file=sys.stderr)
        sys.exit(1)

    ref_path = sys.argv[1]
    hyp_path = sys.argv[2]

    with open(ref_path) as f:
        ref = f.read().strip()
    with open(hyp_path) as f:
        hyp = f.read().strip()

    ref_norm = normalize(ref)
    hyp_norm = normalize(hyp)

    ref_words = ref_norm.split()
    hyp_words = hyp_norm.split()

    if not ref_words:
        print("0 1.0 0 0")
        return

    wer = word_wer(ref_words, hyp_words)
    accuracy = max(0.0, (1.0 - wer)) * 100.0

    print(f"{accuracy:.2f} {wer:.4f} {len(ref_words)} {len(hyp_words)}")


if __name__ == "__main__":
    main()
