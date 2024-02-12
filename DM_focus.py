import glob
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import torch
import numpy as np
import os
from sklearn.linear_model import Ridge
from scipy import stats
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pathlib
# essentially just refactor the sm_pred_mm_2 notebook

# Function to check if all split mutants for a given gene exist in df1
def mutants_exist(row, df1_pairs):
    return all((row['gene'], mutant) in df1_pairs for mutant in row['mutant'].split(":"))

# Function to get the sum of DMS_scores for split mutants of a given gene
def get_dms_score_sum(row, dms_scores):
    return sum(dms_scores.get((row['gene'], mutant), 0) for mutant in row['mutant'].split(":"))

def read_in_PG(query_string):
    # 'ESM_variant_sweep/Protein_Gym/ProteinGym_substitutions/*.csv'
    all_human_files = glob.glob(query_string) #'ProteinGym_substitutions/*')#HUMAN*') 
    filtered_files = [f for f in all_human_files if 'indel' not in f]
    ls_of_df = []
    for file in filtered_files:
        # read in file
        df = pd.read_csv(file)
        df["assay"] = file.split("/")[-1]
        df.mutant = df.mutant.unique()
        # bake in the gene name as a column
        df["gene"] = file.split("/")[-1].split("_")[0] + "_" + file.split("/")[-1].split("_")[1]
        try:
            #print(file.split("/")[1], len(df.index))
            ls_of_df.append(df[["gene", "mutant", "assay", "mutated_sequence", "DMS_score"]])
        except:
            print(file, "did not have mutant name, only sequence") # these needs to be revisited
            print(df.columns)
            pass 
    # then concatenate the dfs -- keep this for later to compare with clinvar mutations 
    all_gene_muts = pd.concat(ls_of_df)
    intersect_set = set(all_gene_muts.assay.unique())
    return intersect_set, all_gene_muts

def get_high_order_constituents(all_gene_muts):
    # separate by single missense or multiple missense
    all_sm = all_gene_muts[~all_gene_muts['mutant'].str.contains(":")]
    all_mm = all_gene_muts[all_gene_muts['mutant'].str.contains(":")]
    # do this with assay instead of gene
    sm_genes = set(all_sm["assay"].unique())
    mm_genes = set(all_mm["assay"].unique())
    # get only those genes that have both sm and mm seqs in same assay
    intersect_set = sm_genes.intersection(mm_genes)
    sm_subset = []
    mm_subset = []
    for assay in intersect_set:
        sm_subset.append(all_sm[all_sm.assay == assay])
        mm_subset.append(all_mm[all_mm.assay == assay])
    sm_subset = pd.concat(sm_subset)
    mm_subset = pd.concat(mm_subset)
    # Create a set of unique (gene, mutant) pairs from df1 for fast lookup
    # global df1_pairs # must be global for mutants_exist to work
    df1_pairs = set(sm_subset[['gene', 'mutant']].itertuples(index=False))
    # Apply the function to check if all split mutants for a given gene exist in df1 from df2
    mm_subset['exists_in_df1'] = mm_subset.apply(lambda row: mutants_exist(row, df1_pairs), axis=1)
    mm_subset = mm_subset[mm_subset['exists_in_df1']]
    # mini_df = (mm_subset[mm_subset.assay == "F7YBW7_MESOW_Ding_2023.csv"])
    # print(len(mini_df[mini_df['mutant'].str.len() == 9].index))
    # print(len(mini_df[mini_df['mutant'].str.len() == 14].index))# only 10 values here, cutoff later
    # print(len(mini_df[mini_df['mutant'].str.len() == 19].index))
    # Create a dictionary with (gene, mutant) as keys and DMS_score as values for fast lookup from df1
    # global dms_scores # needed for get_dms_score_sum to work -- this is dumb
    dms_scores = sm_subset.set_index(['gene', 'mutant'])['DMS_score'].to_dict()
    # Apply the function to each row in df2
    mm_subset['pred_DMS_score'] = mm_subset.apply(lambda row: get_dms_score_sum(row, dms_scores), axis=1) 
    # concat back together
    full = pd.concat((sm_subset, mm_subset))
    return intersect_set, full



def missense_to_WT(AA_str, edit):
    original_AA = edit[0]
    change_AA = edit[-1]
    location = int(edit[1:-1]) -1 # mutations are 1 indexed! -- in this file double check
    # size of prot seq changes between revisions
    if location > len(AA_str):
        return False
    elif AA_str[location] != change_AA: # the indexing is off by one?
        return False
    AA_str = AA_str[:location] + original_AA + AA_str[location+1:]
    #  print([(AA[i], i, editted_AA[i]) for i in range(len(editted_AA)) if editted_AA[i] != AA[i]])
    return AA_str

def add_WT_col(all_sm): # df must have mutated_sequence and mutant cols
    # now we apply this function as a lambda to our df
    all_sm['WT_sequence'] = all_sm.apply(lambda row: missense_to_WT(row['mutated_sequence'], row['mutant']), axis=1)
    return all_sm


def write_wt_fasta(WT_dict, output_file):
    with open(output_file, 'w') as fasta_file:
        for seq_id, sequence in WT_dict.items():
            fasta_file.write(f'>{seq_id}\n')
            fasta_file.write(f'{sequence}\n')

def get_LLR(intersect_set, full, LLR_string, compute_LLR=False):
    # we need to get the correct WT seq each assay
    WT_dict = dict()
    # for each assay, subset the data 
    for assay in intersect_set:
        df = full[full.assay == assay]
        sm = df[~df['mutant'].str.contains(":")]
        num_WT = add_WT_col(sm).WT_sequence.unique()
        assert len(num_WT) == 1, f"multiple WT in assay {assay}"
        WT_dict[assay] = num_WT[0]
    if compute_LLR:
        fasta_name = str(LLR_string).split("/")[-1].split(".")[0] + ".fasta"
        write_wt_fasta(WT_dict, fasta_name)
        # Run the script from the command line
        script_path = '../esm-variants/esm_score_missense_mutations.py'
        os.system(f'python3 {script_path} --input-fasta-file {fasta_name} --output-csv-file {LLR_string}')
        sys.exit()
    # import LLR  
    LLRS = pd.read_csv(LLR_string) #"WT_for_MM_assays.csv")
    LLRS = LLRS.rename(columns={"seq_id":"assay", "mut_name":"mutant", "esm_score":"LLR"}, inplace=False)
    return WT_dict, LLRS

def get_sm_LLR(full, LLRS):
    # now we separate again 
    sm = full[~full['mutant'].str.contains(":")]
    # add LLRS
    sm = (pd.merge(sm, LLRS, on=['assay', 'mutant',]))
    return sm

def main(args):
    query_string =  f"{args.pg_sub_dir}/*.csv"
    
    intersect_set, full = read_in_PG(query_string)

    WT_dict, LLRS = get_LLR(intersect_set, full, args.llr_csv)

    mm_full = full[full['mutant'].str.contains(":")]

    chad = []
    okay = []
    records = []
    # need to get the full df
    sm_full = get_sm_LLR(full, LLRS)
    assert len(sm_full.assay.unique()) == len(intersect_set), f"{len(sm_full.assay.unique())} != {len(intersect_set)}"
    # for each assay, subset the data 
    for assay in intersect_set:
        sm = sm_full[sm_full.assay == assay]
        mm = mm_full[mm_full.assay == assay]
        if mm.index.size != 0:
            # break up each mutation in mm into its constituent parts
            sm_in_mm = mm.mutant.str.split(":", expand=True)
            sm_ls = []
            for column in sm_in_mm.columns:
                sm_ls += list(sm_in_mm[column].unique())
            sm['higher_order'] = sm['mutant'].isin(sm_ls)
        else:
            sm['higher_order'] = False
        # then 
        print(assay, len(sm.mutant.unique()), len(sm[sm.higher_order == True].index))  
        mm["dist_from_WT"] = mm['mutant'].str.count(':') + 1
        # Group the data by 'dist_from_WT' and count the number of rows in each group
        count_dist_from_WT = mm.groupby('dist_from_WT').size()
        # Create a new column 'count_dist_from_WT' with the count for each unique 'dist_from_WT' entry
        mm['eval_size'] = mm['dist_from_WT'].map(count_dist_from_WT)
        print(sm)
        print(mm)
        raise Error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Predict Multimissense Mutations in Protein Gym.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--pg-sub-dir', default= "../ESM_variant_sweep/Protein_Gym/ProteinGym_substitutions/",
                        required=False, type=pathlib.Path, help='Path to Protein Gym Substitution Files.')
    parser.add_argument('--llr-csv', default= "WT_for_SM_filter.csv", #"../WT_for_MM_assays.csv", 
                        required=False, type=pathlib.Path, help="LLR file location.",) 
    # make anoher llr csv with all the sm llrs

    parser.add_argument('--graphs-only', action='store_true', # if graphs-only ignored in input, this var will be True
                        required=False, help="Skip the pipeline and make final graphs only",) 
    parser.add_argument('--only-assay',default=None,  # need to handle default here
                        required=False, type=pathlib.Path, help="Run Pipeline on Only this Assay.",) 
    args = parser.parse_args()

    main(args)