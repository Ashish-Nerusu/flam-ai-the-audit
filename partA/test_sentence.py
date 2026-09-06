import argparse
from pathlib import Path
import tiktoken

BASE_DIR = Path(__file__).resolve().parent
CORPUS_DIR = BASE_DIR / "corpus"

def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def get_stats(lines, encode):
    words = 0
    tokens = 0
    for line in lines:
        words += len(line.split())
        tokens += len(encode(line))
    return words, tokens

def main(args):
    encode = tiktoken.get_encoding("gpt2").encode
    eng_lines = read_lines(args.eng)
    hin_lines = read_lines(args.hin)
    
    eng_w, eng_t = get_stats(eng_lines, encode)
    hin_w, hin_t = get_stats(hin_lines, encode)
    
    print(f"ENG words: {eng_w}, HIN words: {hin_w}")
    print(f"ENG total tokens: {eng_t}, HIN total tokens: {hin_t}")
    print(f"Hindi/English token ratio on parallel sample: {hin_t / eng_t:.2f}x")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eng", default=str(CORPUS_DIR / "starter_eng.txt"))
    parser.add_argument("--hin", default=str(CORPUS_DIR / "starter_hin.txt"))
    args = parser.parse_args()
    main(args)
