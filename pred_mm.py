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

    # filter for human only
    #all_gene_muts = all_gene_muts[all_gene_muts.gene.str.contains("HUMAN")]
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

# stuff to perform chloe hsu augment
aa_to_int = {
    'M':1,
    'R':2,
    'H':3,
    'K':4,
    'D':5,
    'E':6,
    'S':7,
    'T':8,
    'N':9,
    'Q':10, 'C':11,
    'U':12,
    'G':13,
    'P':14,
    'A':15,
    'V':16,
    'I':17,
    'F':18,
    'Y':19,
    'W':20,
    'L':21,
    'O':22, #Pyrrolysine
    'X':23, # Unknown
    'Z':23, # Glutamic acid or GLutamine
    'B':23, # Asparagine or aspartic acid
    'J':23, # Leucine or isoleucine
    'start':24,
    'stop':25,
    '-':26,
}

def aa_seq_to_int(s):
    """
    Return the int sequence as a list for a given string of amino acids
    """
    return [24] + [aa_to_int[a] for a in s] + [25]

def format_seq(seq,stop=False):
    """
    Takes an amino acid sequence, returns a list of integers in the codex of the babbler.
    Here, the default is to strip the stop symbol (stop=False) which would have 
    otherwise been added to the end of the sequence. If you are trying to generate
    a rep, do not include the stop. It is probably best to ignore the stop if you are
    co-tuning the babbler and a top model as well.
    """
    if stop:
        int_seq = aa_seq_to_int(seq.strip())
    else:
        int_seq = aa_seq_to_int(seq.strip())[:-1]
    return int_seq

# take a list of seqs and converts to a batch of seqs
def format_batch_seqs(seqs):
    maxlen = -1
    for s in seqs:
        if len(s) > maxlen:
            maxlen = len(s)
    formatted = []
    for seq in seqs:
        pad_len = maxlen - len(seq)
        padded = np.pad(format_seq(seq), (0, pad_len), 'constant', constant_values=0)
        formatted.append(padded)
    return np.stack(formatted)
# converts ls of seqs -> batch -> one hot encodes seq batch
# is there a way to speed up this function?
# definitely has poor scaling
def seqs_to_onehot(seqs):
    seqs = format_batch_seqs(seqs)
    X = np.zeros((seqs.shape[0], seqs.shape[1]*24), dtype=int)
    for i in range(seqs.shape[1]):
        for j in range(24):
            X[:, i*24+j] = (seqs[:, i] == j)
    return X
# end of chloe hsu augment code

def standardize(sm, mm):
    # standardize the sm, use this scaler to standardize mm
    scaler = StandardScaler()
    sm = scaler.fit_transform(sm)
    mm = scaler.transform(mm)
    return sm, mm

#this  code should be refactored to combine the pred functions. Either compress to one function of use class and inherit
def pred_combo(sm, mm, combos, alpha):
    sm_DMS = sm.DMS_score.values
    mm_DMS = mm.DMS_score.values
    # standardize each column in combos separately
    all_sm, all_mm = [], []
    for column in combos:
        if column in ["layer_21", "layer_33"]:
            # get esm embeddings for all sm
            pre_sm = np.vstack(sm[column].values)
            pre_mm = np.vstack(mm[column].values)
            pre_sm, pre_mm = standardize(pre_sm, pre_mm)
            all_sm.append(pre_sm), all_mm.append(pre_mm)
        elif column == "mutated_sequence":
            # get one hot embeddings for all sm
            seqs = sm[column].values
            pre_sm = seqs_to_onehot(seqs)
            # get one hot embeddings for all sm
            seqs = mm[column].values
            pre_mm = seqs_to_onehot(seqs)
            pre_sm, pre_mm = standardize(pre_sm, pre_mm)
            all_sm.append(pre_sm), all_mm.append(pre_mm)
        else: # this is sum LLR or PLLR
            pre_sm = sm[column].values
            pre_mm = mm[column].values
            pre_sm, pre_mm = standardize(pre_sm.reshape(-1, 1), pre_mm.reshape(-1, 1))
            all_sm.append(pre_sm), all_mm.append(pre_mm)
    # concatenate them
    all_sm = np.concatenate(all_sm, axis=1)
    all_mm = np.concatenate(all_mm, axis=1)
    # train ridge regressor
    lm = Ridge(alpha=alpha)
    lm.fit(all_sm, sm_DMS)
    # use trained regressor to pred DMS
    pred_DMS = lm.predict(all_mm)
    # collect Spearman's, Pearson's
    s, _ = stats.spearmanr(mm_DMS, pred_DMS)
    return s  

# end the pred refactor


def best_chunking_interval(input_string, target_chunks=10):
    """
    Determines the best interval for chunking a string into a specified number of pieces.

    :param input_string: The string to be chunked.
    :param target_chunks: The desired number of chunks (default: 10).
    :return: An integer representing the chunking interval.
    """
    string_length = len(input_string)
    interval = max(1, string_length // target_chunks)  # Ensure the interval is at least 1
    return interval

def prep_heatmap(wt_sequence, df, column):
    # Initialize a matrix for the scores
    num_positions = len(wt_sequence)
    num_possible_mutations = 20  # For each position excluding the WT amino acid
    scores_matrix = np.full((num_positions, num_possible_mutations), np.nan)  # Use NaN for missing data
    
    # All 20 standard amino acids
    all_amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    
    # Mapping from amino acid to index (for columns)
    aa_to_idx = {aa: idx for idx, aa in enumerate(all_amino_acids)}

    min, max = num_positions, 0 
    # process each mutation
    # how is this being done correctly?
    for _, row in df.iterrows():
        mutations = row['mutant'].split(':')
        for mutation in mutations:
            original_aa, position, mutated_aa = mutation[0], int(mutation[1:-1]) - 1, mutation[-1]  # Adjust indexing as needed
            if position < min:
                min = position
            if position > max:
                max = position
            if original_aa == wt_sequence[position]:  # Validate mutation
                col_idx = aa_to_idx.get(mutated_aa)
                if col_idx is not None and original_aa != mutated_aa:
                    scores_matrix[position, col_idx] = row[column]
    #print(scores_matrix.shape)
    #print(min, max)
    # use min and max to slice the score matrix
    scores_matrix = scores_matrix[min:max+1, :]
    #wt_sequence = wt_sequence[min:max+1]
    positions = [i for i in range(min, max+1)]
    #print(scores_matrix.shape)
    return scores_matrix, aa_to_idx, positions, min


def create_ytick_labels(sequence, min):
    """Create y-axis labels at regular intervals."""
    labels = [''] * len(sequence)
    interval = best_chunking_interval(sequence)
    #print(min, interval)
    for i in range(0, len(sequence), interval):
        labels[i] = str(i +min+ 1)  # Adding 1 for 1-based indexing
    return labels


def make_heatmap(scores_matrix, wt_sequence, aa_to_idx, type_name, assay_name, min):
    # Create the heatmap
    plt.figure(figsize=(20, 10))  # Adjust size as needed
    print(scores_matrix.shape)
    ax = sns.heatmap(scores_matrix, #annot=True, 
                fmt=".2f", cmap="viridis",
                yticklabels = create_ytick_labels(wt_sequence, min), # list(wt_sequence),
                xticklabels=sorted(aa_to_idx.keys())
               )
    # use matplotlib.colorbar.Colorbar object
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    plt.title(f"{type_name} for assay: {assay_name}", fontsize = 30)
    plt.xlabel("Substituted Amino Acid", fontsize = 25)
    plt.ylabel("Position in WT Sequence", fontsize = 25)
    #plt.show()
    plt.savefig(f"{type_name}_{assay_name}_heatmap.png") # now need a title
    print(f"{type_name}_{assay_name}_heatmap.png printed")
    
def get_all_heatmaps(WT_dict, assay, sm, WT_PLLRS, pred_ls):
    for pred in pred_ls:
        scores_matrix1, aa_to_idx, positions, min = prep_heatmap(WT_dict[assay], sm, pred)
        make_heatmap(scores_matrix1, positions, aa_to_idx, pred, assay, min)
    print("heatmaps done")
    # use true DMS
    # scores_matrix1, aa_to_idx, positions, min = prep_heatmap(WT_dict[assay], sm, "DMS_score")
    # make_heatmap(scores_matrix1, positions, aa_to_idx, "DMS score", assay, min)
    
    # # pred DMS
    # # scores_matrix2, aa_to_idx, positions = prep_heatmap(WT_dict[assay], all_mm[key], "pred_DMS_score")
    # # make_heatmap(scores_matrix2, positions, aa_to_idx, "pred_DMS_score", assay)
    
    # # sum_LLR
    # scores_matrix3, aa_to_idx, positions, min = prep_heatmap(WT_dict[assay], sm, "sum_LLR")
    # make_heatmap(scores_matrix3, positions, aa_to_idx, "LLR", assay, min)
    # # PLLR, 
    # scores_matrix4, aa_to_idx, positions, min = prep_heatmap(WT_dict[assay], sm, "PLLR")
    # make_heatmap(scores_matrix4, positions, aa_to_idx, "PLLR", assay, min)
    # null return, the heatmaps will be saved

# this function needs to get debugged
def dm_inverted_heatmap(df): # if you make such a function, it must be focused on only one position, and can only work for DM vars
    pass

def make_scatterplot(x,y, colors, xlabel, ylabel, cname, assay):
    plt.scatter(x, y, c=colors)
    plt.xlabel(xlabel, fontsize = 20)
    plt.ylabel(ylabel, fontsize = 20)
    plt.title(f"{xlabel} vs. {ylabel}, assay: {assay}")
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=20)
    cbar.ax.set_ylabel(cname, fontsize=20)
    plt.savefig(f"scatter_{xlabel}_{xlabel}_{cname}_{assay}.png")
    plt.show()

def from_ids_make_df(assay, full, LLRs):
    df = full[full.assay == assay]
    # retrieve all gene and mutant names that were used to construct the fasta
    ids = []
    for index, row in df.iterrows():
        ids.append(f"{row['gene']}_{row['mutant']}")
    print(assay, len(ids))
    new_rows = {"gene":[], "mutant":[], "PLLR":[], "layer_33":[], "layer_21":[]}
    possible_locations = ["0", "1", "2_to_6"] # this might want to be fixed at some point
    for id in ids:
        for j in possible_locations: #range(7):
            if os.path.isfile(f"../sm_pred_mm_{j}/{id}.pt"):# and id not in seen:
                # read the file
                dic = torch.load(f"../sm_pred_mm_{j}/{id}.pt")
                # break label on "_"
                broken_label = dic["label"].split("_")
                new_rows["gene"].append("_".join(broken_label[:2]))
                new_rows["mutant"].append("_".join(broken_label[2:]))
                # get PLLRS
                new_rows["PLLR"].append(dic["PLLRs"])
                # get layer 33
                new_rows["layer_33"].append(dic["mean_representations"][33])
                # get layer 21
                new_rows["layer_21"].append(dic["mean_representations"][21])
    new_df = pd.DataFrame(new_rows)
    #print("ids collected", len(new_df.index))
    assert len(df.index) == len(new_df.index), f"Not all ids captured: {len(new_df.index)} < {len(df.index)}"
    # merge on ids
    fused = (pd.merge(df, new_df, on=['gene', 'mutant',]))
    #print("after fusion", len(fused.index))
    # now we separate again 
    sm = fused[~fused['mutant'].str.contains(":")]
    # add LLRS
    sm = (pd.merge(sm, LLRs, on=['assay', 'mutant',]))
    return new_df, fused, sm

def get_llr_score_sum(row, dms_scores):
    return sum(dms_scores.get((row['assay'], mutant), 0) for mutant in row['mutant'].split(":"))
        
# this isn't working correctly
def get_and_sep_mm(fused, LLRS):
    # get multi missense
    mm = fused[fused['mutant'].str.contains(":")]
    # get the sum LLR
    # Create a dictionary with (gene, mutant) as keys and DMS_score as values for fast lookup from df1
    dms_scores = LLRS.set_index(['assay', 'mutant'])['sum_LLR'].to_dict()
    # Apply the function to each row in df2
    mm["sum_LLR"] = mm.apply(lambda row: get_llr_score_sum(row, dms_scores), axis=1)
    # now we need to sep by the number of mm:
    # Function to count occurrences of ":"
    count_colons = lambda x: x.count(':')  
    # Group by the number of colons
    grouped = mm.groupby(mm['mutant'].apply(count_colons)) 
    # Separate into different DataFrames
    all_mm = {num_colons: group for num_colons, group in grouped}
    return all_mm



def get_spearmans(DMS_scores, pred_ls, estimator_ls, name_ls, assay, all_mm, key, alpha_arr, records, sm):
    for i in range(len(pred_ls)):
        if estimator_ls[i] is None:
            s, _ = stats.spearmanr(DMS_scores, all_mm[key][pred_ls[i]])
            
        else: # (sm, mm, combos, alpha): so first arg is train_set
            s = pred_combo(sm, all_mm[key], pred_ls[i], #["PLLR", "sum_LLR"], 
                           alpha_arr[i])
        print(name_ls[i], alpha_arr[i], key+1, s)
        records.append({"assay": assay, "eval_size": len(all_mm[key].index), "features": name_ls[i], 
                        "dist_from_WT": key+1, "correlation_score":s, "alpha": alpha_arr[i],})
    return records

# now let's do this where we aggregate all previous dist from WT and use it in train.

# if anything, it would be nice for this to be parallelized
def eval_loop(intersect_set, WT_dict, desired, full, LLRS, WT_PLLRS):
    heat_ls = ["DMS_score", "sum_LLR", "PLLR"]
    records = []
    # for each assay, subset the data 
    for assay in intersect_set: # spawn a process for each assay
        if assay in desired:
            # create the df from PG, .pt for representations and LLRS and PLLRS
            new_df, fused, sm = from_ids_make_df(assay, full, LLRS) 
            # add normalize with WT
            appropriate_WT_PLLR = (WT_PLLRS[WT_PLLRS.assay == assay].PLLR.values[0])
            # Index(['gene', 'mutant', 'assay', 'mutated_sequence', 'DMS_score',
            # 'exists_in_df1', 'pred_DMS_score', 'PLLR', 'layer_33', 'layer_21']
            # print(sm["mutated_sequence"].values[:2])
            # print("sm['PLLR']", sm["PLLR"].values[:2])
            # print("appropriate_WT_PLLR", appropriate_WT_PLLR)
            # raise Error
            # try to apply the WT PLLR normalization here
            sm["PLLR"] = sm.PLLR - appropriate_WT_PLLR
            # make all the SM heatmaps for DMS, LLR, PLLR
            get_all_heatmaps(WT_dict, assay, sm, WT_PLLRS, heat_ls)
            # need to normalilze PLLR for mm as well
            fused["PLLR"] = fused.PLLR - appropriate_WT_PLLR
            # separate the MM df for each dataframe by number of missense
            all_mm = get_and_sep_mm(fused, LLRS)
            # record sm_pred_mm performance
            if len(all_mm.keys()) > 0 : #1:
                for key in all_mm.keys():  
                    if len(all_mm[key].index) > 10:#and key == 1:
                        print(assay, key+1, len(all_mm[key].index))
                        # pred DMS and sum LLR are empty right now
                        
                        # first sum the DMS to pred MM
                        pred_ls = ["pred_DMS_score",  "sum_LLR", "PLLR"] + 2*[
                                   ["PLLR", "sum_LLR"], 
                                   ["mutated_sequence"],
                                   ["layer_33"],
                                   ["layer_21"],
                                   ["mutated_sequence", "sum_LLR"], 
                                   ["layer_33", "sum_LLR"], 
                                   ["layer_21", "sum_LLR"], 
                                   ["layer_21", "layer_33", "sum_LLR"],
                                   ["mutated_sequence","layer_21", "layer_33", "sum_LLR"],
                                   ["mutated_sequence","layer_21", "layer_33", "sum_LLR", "PLLR"],
                        ]
                                   
                        estimator_ls = [None, None, None]+ 2*10*[Ridge]
                        name_ls = ["sum_DMS", "sum_LLR", "PLLR"] + 2*[
                            "PLLR+sum_LLR",
                            "one_hot", 
                            "layer_33", 
                            "layer_21", 
                            "one_hot+sum_LLR", 
                            "layer_33+sum_LLR", 
                            "layer_21+sum_LLR",
                            "layer_21+layer_33+sum_LLR", 
                            "one_hot+layer_21+layer_33+sum_LLR", 
                            "one_hot+layer_21+layer_33+sum_LLR+PLLR"
                        ]
                        alpha_arr = ["N/A", "N/A", "N/A"] + 10*[0] + 10*[100]
                        # so this version uses only sm for train
                        records = get_spearmans(all_mm[key].DMS_score, pred_ls, estimator_ls, name_ls, assay, all_mm, key, alpha_arr, records, sm)
                        # now do the exact same, but sm + all previous mm for train: just need to change the sm arg
                        if key > 1:
                            name_ls = [f"{name}_redux" for name in name_ls]
                            records = get_spearmans(all_mm[key].DMS_score, pred_ls, estimator_ls, name_ls, assay, all_mm, key, alpha_arr, records, 
                                                    pd.concat([sm] + [all_mm[k] for k in all_mm.keys() if k < key]))
                        
                        # now we need to collect the other ones

    #Convert the list of records into a DataFrame
    all_records = pd.DataFrame(records)
    print(all_records)
    all_records.to_csv("MM_Assay_splits.csv")

def results_bargraph(group_data, title, figname):
    plt.figure(figsize=(25, 6))
    sns.barplot(x='features', y='correlation_score', hue='alpha', data=group_data)
    plt.title(title) #f'Assay: {assay}, Distance from WT: {dist_from_WT}, Evaluation Size:{eval_size}')
    plt.xlabel('Features')
    plt.ylabel('Correlation')
    plt.legend(title='Alpha')
    plt.savefig(figname) #f"SM_pred_{assay}_{dist_from_WT}.png")# show()
    plt.close()

# I want another graph: one where for each feature type, we get a line plot of how the correlation changes with distance from WT
def results_lineplot(group_data, title, figname):
    print("group_data", group_data)
    color_mapping = {
        # unsupervised
        "sum_LLR": "red",
        "PLLR": "pink",
        # one hot without embeddings
        "sum_DMS": "blue",
        "one_hot": "darkblue",
        "one_hot+sum_LLR": "lightblue",
        'PLLR+sum_LLR': "purple",
        # ESM layer or ESM layer + sum LLR
        "layer_21": "green",
        "layer_33": "yellow",
        "layer_21+sum_LLR": "lightgreen",
        "layer_33+sum_LLR": "orange",
        "21+33+LLR": "darkgreen",
        # wambo combos
        "oh+21+33+LLR": "brown",
        "oh+21+33+LLR+PLLR": "black",

    }
    color_order = ["sum_DMS", "one_hot", "sum_LLR", "PLLR", "one_hot+sum_LLR", "PLLR+sum_LLR", 
                   "layer_21", "layer_33", "layer_21+sum_LLR", "layer_33+sum_LLR", "21+33+LLR", 
                   "oh+21+33+LLR", "oh+21+33+LLR+PLLR"]
    
    color_order = [f"{key}_redux" for key in color_order]
    color_mapping = ({f"{key}_redux": value for key, value in color_mapping.items()})
    print(color_mapping)
    # create the combo of dist_form_WT and eval_size
    group_data["X-axis"] = group_data.apply(lambda row: f"{row['dist_from_WT']}, {row['eval_size']}", axis=1)
    print(group_data["X-axis"])
    plt.figure(figsize=(20, 8))
    ax = sns.lineplot(x ='X-axis', #x='dist_from_WT',
                       y='correlation_score',
                 hue = 'features', style = 'alpha',
                 palette=color_mapping,
                 hue_order=color_order,
                #hue='alpha', 
                 data=group_data)
    
    plt.title(title, fontsize=30) #f'Assay: {assay}, Distance from WT: {dist_from_WT}, Evaluation Size:{eval_size}')
    plt.xlabel('Distance from WT, Number of Variants', fontsize=30)
    plt.ylabel('Correlation', fontsize=30)
    plt.legend(title='Features and Alpha', fontsize=24)

    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    print(figname)
    plt.savefig(figname, bbox_inches='tight') #f"SM_pred_{assay}_{dist_from_WT}.png")# show()
    plt.close()

def plot_all_results(results_path):
    # some assays come up twice
    all_assays = pd.read_csv(results_path) #"MM_Assay_splits.csv")
    print(all_assays)
    #all_assays["features"] = (all_assays.features.str.split('+').apply(lambda x: [element.split("_")[-1] for element in x]).apply(lambda x: '+'.join(x)))#.str.split('_'))
    all_assays['alpha'] = all_assays['alpha'].replace(np.nan, 'Not Available')
    # will need to delete duplicate assays
    all_assays = (all_assays[all_assays.assay != "CAPSD_AAV2S_Sinai_2021.csv"])

    # now we need to shorten the names of sum of features
    all_assays['features'] = all_assays['features'].replace({
        'one_hot+layer_21+layer_33+sum_LLR+PLLR': 'oh+21+33+LLR+PLLR',
        'one_hot+layer_21+layer_33+sum_LLR': 'oh+21+33+LLR',
        'layer_21+layer_33+sum_LLR': '21+33+LLR',
        'one_hot+layer_21+layer_33+sum_LLR+PLLR_redux': 'oh+21+33+LLR+PLLR_redux',
        'one_hot+layer_21+layer_33+sum_LLR_redux': 'oh+21+33+LLR_redux',
        'layer_21+layer_33+sum_LLR_redux': '21+33+LLR_redux'
    })

    # Group the data by assay and dist_from_WT
    grouped = all_assays.groupby(['assay', 'dist_from_WT', 'eval_size'])
    # for (assay, dist_from_WT, eval_size), group_data in grouped:
    #     if dist_from_WT == 10:
    #         print(assay, dist_from_WT, eval_size)
    #     results_bargraph(group_data,
    #                      f'Assay: {assay}, Distance from WT: {dist_from_WT}, Evaluation Size:{eval_size}', 
    #                      f"SM_pred_{assay}_{dist_from_WT}.png")

    # lineplot for each feature type for each assay
    # grouped = all_assays.groupby(['assay', 'features'])
    # for (assay, feature), group_data in grouped:
    #     if len(group_data.dist_from_WT.unique()) > 1:
    #         results_lineplot(group_data,
    #                         f'Assay: {assay}, Feature: {feature}', 
    #                         f"SM_pred_{assay}_{feature}.png")
    #         print(f"SM_pred_{assay}_{feature}.png")

    # lineplot for each feature type for all assays
    # if we could do this for all features at once that would be preferrable
    # grouped = all_assays.groupby( ['alpha'])#[ 'features'])
    # for ( alpha), group_data in grouped: # feature
    #     if len(group_data.dist_from_WT.unique()) > 1:
    #         results_lineplot(group_data,
    #                         f'All Assays, Alpha: {alpha[0]}', # Feature: {feature[0]}', 
    #                         f"SM_pred_{alpha[0]}_all_assays.png")
    #         print(f"SM_pred_{alpha[0]}_all_assays.png")

    # graph only those with dist from WT < 15
    # results_lineplot(all_assays[(all_assays.dist_from_WT < 15) & (all_assays['features'].str.contains('redux'))],
    #                         f'All Assays, All Features',
    #                         f"SM_pred_all_features_all_assays.png")
    # do this but only for one assay in question:
    # we wil do this for 3 assays in question (all the ones that get 14 dist from WT)
    desired_assays = ["PHOT_CHLRE_Chen_2023_multiples.csv"]
    # ["CAPSD_AAV2S_Sinai_substitutions_2021.csv",
    #     "HIS7_YEAST_Pokusaeva_2019.csv", 
    #     "PHOT_CHLRE_Chen_2023_multiples.csv",
    #     "CTHL3_BOVIN_Koch_2022.csv",
    #     "D7PM05_CLYGR_Somermeyer_2022.csv",
    #     "GFP_AEQVI_Sarkisyan_2016.csv",
    #     "H3JQU7_ENTQU_Poelwijk_2019.csv"
    # ]
    for assay in desired_assays:
        results_lineplot(all_assays[(all_assays.assay == assay) & (all_assays['features'].str.contains('redux'))],
                            f'Assay: {assay}, All Features',
                            f"SM_pred_all_features_{assay}.png")
    # now we make one for each distance from wildtype
    grouped = all_assays.groupby(['dist_from_WT'])
    #print(grouped)
    
    for (dist_from_WT), group_data in grouped:
        print(dist_from_WT[0], len(group_data.assay.unique()), sum(group_data.eval_size.unique()))
        results_bargraph(group_data,
                    f'Distance from WT: {dist_from_WT[0]}, Number of Assays: {len(group_data.assay.unique())}, Total Evaluation Size: {sum(group_data.eval_size.unique())}',
                    f"SM_pred_{dist_from_WT[0]}_all_assays.png")
                    

def main(args):
    if not args.graphs_only:
        query_string =  f"{args.pg_sub_dir}/*.csv" #'../ESM_variant_sweep/Protein_Gym/ProteinGym_substitutions/*.csv'
        intersect_set, full = read_in_PG(query_string)
        
            # desired assays:
        if args.only_assay is None: # need some default here
            desired = intersect_set 
        else:
            desired =  [args.only_assay] # force only one assay for now
            #[
                    # "Q8WTC7_9CNID_Somermeyer_2022.csv",
                    # "H3JQU7_ENTQU_Poelwijk_2019.csv",
                    # "Q6WV13_9MAXI_Somermeyer_2022.csv",
                    # "GFP_AEQVI_Sarkisyan_2016.csv",
                    #"PHOT_CHLRE_Chen_2023_multiples.csv"
                    # "CBPA2_HUMAN_Rocklin_2023_1O6X.csv",
                #  "RASK_HUMAN_Weng_2022_binding-RAF1.csv", 
                #"RASK_HUMAN_Weng_2022_abundance.csv", 
                #"AMFR_HUMAN_Rocklin_2023_4G3O.csv",
                # "HIS7_YEAST_Pokusaeva_2019.csv" , 
        #           "CAPSD_AAV2S_Sinai_substitutions_2021.csv"
        #]
        LLR_string, WT_PLLR_string = args.llr_csv, args.wt_pllr_dir #"../WT_for_MM_assays.csv", "../WT_for_MM_assays_redux/*.pt"#WT_for_MM_assays_extra/*.pt"
        WT_dict, LLRS, WT_PLLRS = get_LLR_and_WT_PLLR(intersect_set, full, LLR_string, WT_PLLR_string)
        eval_loop(intersect_set, WT_dict, desired, full, LLRS, WT_PLLRS)
    # results location is hardcoded at the moment
    plot_all_results("MM_Assay_splits.csv") #args.results_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Predict Multimissense Mutations in Protein Gym.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--pg-sub-dir', default= "../ESM_variant_sweep/Protein_Gym/ProteinGym_substitutions/",
                        required=False, type=pathlib.Path, help='Path to Protein Gym Substitution Files.')
    parser.add_argument('--llr-csv', default="../WT_for_MM_assays.csv", 
                        required=False, type=pathlib.Path, help="LLR file location.",) 
    parser.add_argument('--wt-pllr-dir',default= "../WT_for_MM_assays_redux/*.pt",  
                        required=False, help="WT PLLR file location.",) 
    parser.add_argument('--graphs-only', action='store_true', # if graphs-only ignored in input, this var will be True
                        required=False, help="Skip the pipeline and make final graphs only",) 
    parser.add_argument('--only-assay',default=None,  # need to handle default here
                        required=False, type=pathlib.Path, help="Run Pipeline on Only this Assay.",) 
    args = parser.parse_args()

    main(args)
