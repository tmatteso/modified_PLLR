import pandas as pd
import numpy as np
import argparse
import glob
import torch 

def read_fasta_file(file_path):
        """
        Read a fasta file and return a dataframe with columns for name and sequence.
        
        Parameters:
        file_path (str): Path to the fasta file.
        
        Returns:
        pandas.DataFrame: Dataframe with columns for name and sequence.
        """
        names, sequences = [], []
        with open(file_path, 'r') as file:
            name = None
            sequence = ''
            for line in file:
                line = line.strip()
                if line.startswith('>'):
                    if name is not None:
                        names.append(name)
                        sequences.append(sequence)
                    name = line[1:]
                    sequence = ''
                else:
                    sequence += line
            if name is not None:
                names.append(name)
                sequences.append(sequence)
        df = pd.DataFrame({'name': names, 'mut_seq': sequences})
        return df 

def read_from_pt(fasta, dir_loc):
    # needs the input fasta too to get the sequence back
    df = read_fasta_file(fasta)
    # add the gene and mutant columns
    df["gene"] = df["name"].apply(lambda x: x.split("_")[0])
    df["mutant"] = df["name"].apply(lambda x: x.split("_")[1])
    # get the pt files
    files = glob.glob(f"{dir_loc}/*.pt")
    new_rows = {"gene":[], "mutant":[], "esm_score":[], }
    for file in files:
        # read the file
        dic = torch.load(file)
        # break label on "_"
        broken_label = dic["label"].split("_")
        new_rows["gene"].append("_".join(broken_label[:2]))
        new_rows["mutant"].append("_".join(broken_label[2:]))
        # get PLLRS
        new_rows["esm_score"].append(dic["PLLRs"])
    new_df = pd.DataFrame(new_rows)
    #print("ids collected", len(new_df.index))
    assert len(df.index) == len(new_df.index), f"Not all ids captured: {len(new_df.index)} < {len(df.index)}"
    # merge on ids
    fused = (pd.merge(df, new_df, on=['gene', 'mutant',]))
    
    return fused

def main(args):
    # load in Nadav's modified results
    nadav_csv = pd.read_csv(args.nadav_csv)[["mut_seq", "esm_score"]]
    # load in my results
    extract_csv = pd.read_csv(args.extract_csv)[["mut_seq", "esm_score"]] 
    # I need to account for the other results type and see any discrepancies
    pt_PLLRs = read_from_pt(args.extract_fasta, args.extract_pt)
    # Join the two dataframes on the mut_seq column
    merged_df = nadav_csv.merge(extract_csv, on="mut_seq")
    # join with the other results
    merged_df = merged_df.merge(pt_PLLRs, on="mut_seq")
    # Print the new dataframe
    print(merged_df)

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='covert bulk fasta extraction format to nadav script format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--nadav-csv', required=True,)
    parser.add_argument('--extract-csv', required=True,)
    parser.add_argument('--extract-fasta', required=True,)
    parser.add_argument('--extract-pt', required=True,)
    args = parser.parse_args()

    main(args)