import pandas as pd
import numpy as np
import argparse


def main(args):
    # load in Nadav's modified results
    nadav_csv = pd.read_csv(args.nadav_csv)[["mut_seq", "esm_score"]]
    # load in my results
    extract_csv = pd.read_csv(args.extract_csv)[["mut_seq", "esm_score"]] 
    
    # Set mut_seq as index for faster lookup
    nadav_csv.set_index("mut_seq", inplace=True)
    extract_csv.set_index("mut_seq", inplace=True)
    
    # Check consistency using vectorized operations
    is_consistent = np.isclose(nadav_csv["esm_score"], extract_csv["esm_score"], atol=0.001)
    failed_index = is_consistent[~is_consistent].index
    
    if len(failed_index) == 0:
        print("All rows are consistent.")
    else:
        print(f"Rows {', '.join(map(str, failed_index))} failed the consistency check.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='covert bulk fasta extraction format to nadav script format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--nadav-csv', dest='input_fasta_file', required=True, metavar='/path/to/input_mutations.fasta', help='Path to the input CSV file with the protein mutations to calculate ESM scores for.')
    parser.add_argument('--extract-csv', dest='output_csv_file', required=True, metavar='./esm_multi_residue_effect_scores.csv', help='Path to save the output CSV file.')
    args = parser.parse_args()

    main(args)