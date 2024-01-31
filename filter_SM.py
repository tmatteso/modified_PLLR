# screen the sm for all assays by their scatterplot, R^2 and the LLR Spearman's Corre

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
from sklearn.metrics import r2_score


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

def get_LLR(intersect_set, full, LLR_string):
    # we need to get the correct WT seq each assay
    WT_dict = dict()
    # for each assay, subset the data 
    for assay in intersect_set:
        df = full[full.assay == assay]
        sm = df[~df['mutant'].str.contains(":")]
        num_WT = add_WT_col(sm).WT_sequence.unique()
        assert len(num_WT) == 1, f"multiple WT in assay {assay}"
        WT_dict[assay] = num_WT[0]
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


# we will come back and color the dots by their presence in higher order mutations
def make_scatterplot(x, y, xlabel, ylabel, assay):
    plt.scatter(x, y)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.title(f"{xlabel} vs. {ylabel}, assay: {assay}")
    
    # Add regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    plt.plot(x, intercept + slope * x, color='red')
    
    # Calculate and display R-squared value
    r_squared = r2_score(y, intercept + slope * x)
    plt.text(0.05, 0.95, f"R-squared: {r_squared:.2f}", transform=plt.gca().transAxes, fontsize=12)
    
    # cbar = plt.colorbar()
    # cbar.ax.tick_params(labelsize=20)
    # cbar.ax.set_ylabel(cname, fontsize=20)
    plt.savefig(f"scatter_{xlabel}_{xlabel}_{assay}.png")
    plt.show()


# if anything, it would be nice for this to be parallelized
def eval_loop(intersect_set, desired, full, LLRS, output_csv):
    records = []
    # need to get the full df
    sm_full = get_sm_LLR(full, LLRS)
    # for each assay, subset the data 
    for assay in intersect_set: # spawn a process for each assay
        if assay in desired:
            sm = sm_full[sm_full.assay == assay]
            print(sm_full)
            s, _ = stats.spearmanr(sm.DMS_score, sm.LLR)
            records.append({"assay": assay, "eval_size": len(sm.index), "features": "LLR", 
                        "dist_from_WT": 1, "correlation_score":s, "alpha": "N/A",})
            xlabel, ylabel = "DMS_score", "LLR"
            make_scatterplot(sm.DMS_score,sm.LLR, xlabel, ylabel, assay)
            raise Error


    #Convert the list of records into a DataFrame
    all_records = pd.DataFrame(records)
    print(all_records)
    all_records.to_csv(output_csv) #"MM_Assay_splits.csv")

def main(args):
    output_csv = "SM_filter_Assay_splits.csv"
    if not args.graphs_only:
        # desired assays:
        if args.only_assay is None:
            query_string =  f"{args.pg_sub_dir}/*.csv"
            intersect_set, full = read_in_PG(query_string)
            WT_dict, LLRS = get_LLR(intersect_set, full, args.llr_csv)
            eval_loop(intersect_set, intersect_set, full, LLRS, output_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Predict Multimissense Mutations in Protein Gym.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--pg-sub-dir', default= "../ESM_variant_sweep/Protein_Gym/ProteinGym_substitutions/",
                        required=False, type=pathlib.Path, help='Path to Protein Gym Substitution Files.')
    parser.add_argument('--llr-csv', default="../WT_for_MM_assays.csv", 
                        required=False, type=pathlib.Path, help="LLR file location.",) 
    parser.add_argument('--graphs-only', action='store_true', # if graphs-only ignored in input, this var will be True
                        required=False, help="Skip the pipeline and make final graphs only",) 
    parser.add_argument('--only-assay',default=None,  # need to handle default here
                        required=False, type=pathlib.Path, help="Run Pipeline on Only this Assay.",) 
    args = parser.parse_args()

    main(args)
