import os
from datasets import load_dataset
import random

def main():
    # We will use Muennighoff/flores200 (a mirror of facebook/flores)
    langs = {
        "eng": "eng_Latn",
        "hin": "hin_Deva",
        "kan": "kan_Knda",
        "tel": "tel_Telu"
    }
    
    os.makedirs("partA/corpus", exist_ok=True)
    
    num_samples = 500
    random.seed(42)
    
    for short_code, hf_code in langs.items():
        print(f"Downloading {hf_code}...")
        ds = load_dataset("Muennighoff/flores200", hf_code, split="dev", trust_remote_code=True)
        sentences = ds["sentence"]
        
        sample_sentences = sentences[:num_samples]
        
        out_path = f"partA/corpus/{short_code}.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            for s in sample_sentences:
                f.write(s.strip() + "\n")
                
        print(f"Saved {len(sample_sentences)} sentences to {out_path}")

if __name__ == "__main__":
    main()
