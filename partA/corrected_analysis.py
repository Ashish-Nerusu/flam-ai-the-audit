import argparse
from pathlib import Path
import os
from transformers import AutoTokenizer
import tiktoken

BASE_DIR = Path(__file__).resolve().parent
CORPUS_DIR = BASE_DIR / "corpus"

def load_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]

def get_stats(lines, encode_fn):
    words, chars, tokens = 0, 0, 0
    for line in lines:
        words += len(line.split())
        chars += len(line.encode("utf-8")) # UTF-8 bytes, not characters
        tokens += len(encode_fn(line))
    return words, chars, tokens

def main(args):
    langs = ["eng", "hin", "kan", "tel"]
    corpus = {}
    for lang in langs:
        corpus[lang] = load_lines(os.path.join(args.corpus_dir, f"{lang}.txt"))

    tokenizers = {
        "gpt2": tiktoken.get_encoding("gpt2").encode,
        "xlm-roberta": AutoTokenizer.from_pretrained("xlm-roberta-base").encode
    }

    for t_name, encode_fn in tokenizers.items():
        print(f"\nTokenizer: {t_name}")
        print("-" * 50)
        print(f"{'Lang':<6} | {'Tok/Word':<10} | {'Tok/Byte':<10} | {'Tok/Sentence (Workload Ratio)':<22}")
        print("-" * 50)
        
        eng_tokens_per_sentence = 0
        
        for lang in langs:
            lines = corpus[lang]
            words, bytes_, tokens = get_stats(lines, encode_fn)
            
            tok_per_word = tokens / words
            tok_per_byte = tokens / bytes_
            tok_per_sentence = tokens / len(lines)
            
            if lang == "eng":
                eng_tokens_per_sentence = tok_per_sentence
            
            print(f"{lang:<6} | {tok_per_word:<10.2f} | {tok_per_byte:<10.3f} | {tok_per_sentence:<22.1f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_dir", default=str(CORPUS_DIR))
    args = parser.parse_args()
    main(args)
