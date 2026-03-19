import os
import pandas as pd
import numpy as np
import argparse
import gzip
import json
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Calculate Daily GEP Index using Text and Meta-data.")
    parser.add_argument("--dict_path", type=str, required=True, help="Path to geoeconomic_dictionary.csv")
    parser.add_argument("--text_dir", type=str, required=True, help="Path to DATA1 (text files)")
    parser.add_argument("--meta_dir", type=str, required=True, help="Path to INFO_DATA1 (meta files)")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save final index")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load dictionary weights
    dict_df = pd.read_csv(args.dict_path)
    T_dict = dict(zip(dict_df['word'], dict_df['weight']))
    relevant_words = set(T_dict.keys())

    # Find text files
    text_files = sorted([f for f in os.listdir(args.text_dir) if f.endswith('.txt.gz')])
    
    all_daily_results = []

    for txt_file in text_files:
        # Identify corresponding meta file (e.g., rtrs_1996_meta.jsonl.gz)
        year = txt_file.split('_')[1]
        meta_file = f"rtrs_{year}_meta.jsonl.gz"
        
        txt_path = os.path.join(args.text_dir, txt_file)
        meta_path = os.path.join(args.meta_dir, meta_file)
        
        if not os.path.exists(meta_path):
            print(f"[SKIP] Meta file not found for {txt_file}")
            continue

        print(f"[INFO] Processing Year: {year}")
        daily_data = []

        try:
            with gzip.open(txt_path, 'rt', encoding='utf-8') as f_txt, \
                 gzip.open(meta_path, 'rt', encoding='utf-8') as f_meta:
                
                # Iterate through both files line-by-line (synchronized)
                for txt_line, meta_line in tqdm(zip(f_txt, f_meta)):
                    meta_obj = json.loads(meta_line)
                    # Extract date from firstCreated (YYYY-MM-DD)
                    date_str = meta_obj['firstCreated'][:10]
                    
                    text = txt_line.strip()
                    words = text.split()
                    total_words = len(words)
                    
                    if total_words == 0:
                        continue
                    
                    # Scoring logic with frequency capping at 4
                    weighted_sum = 0.0
                    found_words = [w for w in words if w in relevant_words]
                    counts = {}
                    for w in found_words:
                        counts[w] = min(counts.get(w, 0) + 1, 4)
                    
                    for w, count in counts.items():
                        weighted_sum += (count * T_dict[w])
                    
                    # Article score normalized by length
                    score = weighted_sum / total_words
                    daily_data.append({'date': date_str, 'score': score})

            if daily_data:
                df_year = pd.DataFrame(daily_data)
                # Daily Average (Intensity) 
                daily_avg = df_year.groupby('date')['score'].mean().reset_index()
                # Daily Sum (Volume variant) [cite: 1952]
                daily_sum = df_year.groupby('date')['score'].sum().reset_index()
                daily_sum.rename(columns={'score': 'score_volume'}, inplace=True)
                
                all_daily_results.append(pd.merge(daily_avg, daily_sum, on='date'))

        except Exception as e:
            print(f"[ERROR] Failed year {year}: {e}")

    # Final consolidation
    final_index = pd.concat(all_daily_results).sort_values('date')
    output_path = os.path.join(args.output_dir, "GEP_Daily_Index.csv")
    final_index.to_csv(output_path, index=False)
    print(f"[SUCCESS] Daily GEP Index saved to {output_path}")

if __name__ == "__main__":
    main()