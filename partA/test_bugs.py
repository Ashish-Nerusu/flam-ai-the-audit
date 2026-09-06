import argparse
from pathlib import Path
import tiktoken
import unicodedata
import os

BASE_DIR = Path(__file__).resolve().parent
CORPUS_DIR = BASE_DIR / "corpus"

def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def analyze_v0(lines, encode, lower=True):
    per_line_fertility = []
    for line in lines:
        if lower:
            line = line.lower()
        tokens = encode(line)
        words = line.split(" ")
        per_line_fertility.append(len(tokens) / len(words))
    return sum(per_line_fertility) / len(per_line_fertility)

def analyze_v1_micro(lines, encode, lower=True):
    total_tokens = 0
    total_words = 0
    for line in lines:
        if lower:
            line = line.lower()
        total_tokens += len(encode(line))
        total_words += len(line.split(" "))
    return total_tokens / total_words

def main(args):
    encode = tiktoken.get_encoding("gpt2").encode
    
    eng_lines = read_lines(args.eng)
    hin_lines = read_lines(args.hin)
    
    print("--- Original (Macro Average) ---")
    print(f"ENG: {analyze_v0(eng_lines, encode):.3f}")
    print(f"HIN: {analyze_v0(hin_lines, encode):.3f}")
    
    print("\n--- Corrected (Micro Average) ---")
    print(f"ENG: {analyze_v1_micro(eng_lines, encode):.3f}")
    print(f"HIN: {analyze_v1_micro(hin_lines, encode):.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eng", default=str(CORPUS_DIR / "starter_eng.txt"))
    parser.add_argument("--hin", default=str(CORPUS_DIR / "starter_hin.txt"))
    args = parser.parse_args()
    main(args)
