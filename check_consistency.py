import pandas as pd
import numpy as np
import argparse


def main(args):
    # load in Nadav's modified results
    nadav_csv = pd.read_csv(args.nadav_csv)[["mut_seq", "esm_score"]]
    # load in my results
    extract_csv = pd.read_csv(args.extract_csv)[["mut_seq", "esm_score"]] 
    
    # Join the two dataframes on the mut_seq column
    merged_df = nadav_csv.merge(extract_csv, on="mut_seq")
    
    # Print the new dataframe
    print(merged_df)
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='covert bulk fasta extraction format to nadav script format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--nadav-csv', required=True, metavar='/path/to/input_mutations.fasta', help='Path to the input CSV file with the protein mutations to calculate ESM scores for.')
    parser.add_argument('--extract-csv', required=True, metavar='./esm_multi_residue_effect_scores.csv', help='Path to save the output CSV file.')
    args = parser.parse_args()

    main(args)