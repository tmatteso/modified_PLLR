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

def get_LLR_and_WT_PLLR(intersect_set, full, LLR_string, WT_PLLR_string):
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
    LLRS = LLRS.rename(columns={"seq_id":"assay", "mut_name":"mutant", "esm_score":"sum_LLR"}, inplace=False)
    
    # WT PLLR
    all_files = glob.glob(WT_PLLR_string) #"WT_for_MM_assays_extra/*.pt")
    new_rows = {"assay":[], "PLLR":[], }#"layer_33":[], "layer_21":[]}
    for file in all_files:
        # Load the PyTorch tensor
        dic = torch.load(file)
        new_rows["assay"].append(dic["label"])
        # collect PLLRs separately
        new_rows["PLLR"].append(dic["PLLRs"])
    
    WT_PLLRS = pd.DataFrame(new_rows)
    return WT_dict, LLRS, WT_PLLRS



