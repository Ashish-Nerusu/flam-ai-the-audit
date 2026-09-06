import argparse
import sys
from pathlib import Path
import tiktoken

BASE_DIR = Path(__file__).resolve().parent
CORPUS_DIR = BASE_DIR / "corpus"

def read_lines(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def analyze(lines, encode, split_char=None):
    per_line_fertility = []
    per_line_tpc = []
    for line in lines:
        line = line.lower()
        tokens = encode(line)
        if split_char is not None:
            words = line.split(split_char)
        else:
            words = line.split()
        chars = len(line)
        per_line_fertility.append(len(tokens) / len(words))
        per_line_tpc.append(len(tokens) / chars)
    n = len(per_line_fertility)
    return sum(per_line_fertility) / n, sum(per_line_tpc) / n

def main(args):
    encode = tiktoken.get_encoding("gpt2").encode
    eng_lines = read_lines(args.eng)
    hin_lines = read_lines(args.hin)
    
    print("--- Original split(' ') ---")
    fert_eng, tpc_eng = analyze(eng_lines, encode, " ")
    print(f"ENG fertility: {fert_eng:.3f}, tok/char: {tpc_eng:.3f}")
    fert_hin, tpc_hin = analyze(hin_lines, encode, " ")
    print(f"HIN fertility: {fert_hin:.3f}, tok/char: {tpc_hin:.3f}")
    
    print("\n--- Corrected split() ---")
    fert_eng2, tpc_eng2 = analyze(eng_lines, encode, None)
    print(f"ENG fertility: {fert_eng2:.3f}, tok/char: {tpc_eng2:.3f}")
    fert_hin2, tpc_hin2 = analyze(hin_lines, encode, None)
    print(f"HIN fertility: {fert_hin2:.3f}, tok/char: {tpc_hin2:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eng", default=str(CORPUS_DIR / "starter_eng.txt"))
    parser.add_argument("--hin", default=str(CORPUS_DIR / "starter_hin.txt"))
    args = parser.parse_args()
    main(args)
