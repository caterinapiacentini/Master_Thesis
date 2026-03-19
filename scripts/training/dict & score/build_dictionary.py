import os
import pandas as pd
import argparse
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description="Consolidate GTM topic CSVs into a global dictionary.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to GTM results")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save final dictionary")
    parser.add_argument("--num_topics", type=int, default=6, help="Total number of topics (Q)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Dictionary to sum normalized weights: Weight_global = (1/Q) * sum(local_weights)
    word_weight_sums = defaultdict(float)

    # Identify all topic CSVs (e.g., topic_Tariffs.csv)
    csv_files = [f for f in os.listdir(args.input_dir) if f.startswith("topic_") and f.endswith(".csv")]
    
    if not csv_files:
        print(f"No topic CSVs found in {args.input_dir}")
        return

    for file_name in csv_files:
        file_path = os.path.join(args.input_dir, file_name)
        # GTM_6.py output format has 'weight' column and word as index
        df = pd.read_csv(file_path, index_col=0)
        
        # Local Normalization: Scaling weight to [0,1] range as per Paper logic (Section 2.1)
        # Ensures one subtopic doesn't dominate due to scale differences
        max_val = df['weight'].max()
        if max_val > 0:
            df['weight'] = df['weight'] / max_val
        
        for word, row in df.iterrows():
            word_weight_sums[str(word).strip()] += float(row['weight'])

    # Aggregate using the Mean Logic (Equation 10 in paper)
    final_list = []
    for word, total_weight in word_weight_sums.items():
        # Global weight is the average importance across all Q sub-dimensions
        mean_weight = total_weight / args.num_topics
        final_list.append({'word': word, 'weight': mean_weight})

    # Sort descending: words closer to the combined center appear first
    df_final = pd.DataFrame(final_list).sort_values(by='weight', ascending=False)

    output_path = os.path.join(args.output_dir, "geoeconomic_dictionary.csv")
    df_final.to_csv(output_path, index=False)
    print(f"Success: Dictionary created at {output_path}")

if __name__ == "__main__":
    main()