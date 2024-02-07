# screen the sm for all assays by their scatterplot, R^2 and the LLR Spearman's Corre

import glob
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import torch
import numpy as np
from sklearn.linear_model import Ridge
from scipy import stats
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pathlib
from sklearn.metrics import r2_score
import os
import sys


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

# okay so now we need a new column in the df that notes if it appears in higher order mutations
# then simply call scatter again with a different color

# we will come back and color the dots by their presence in higher order mutations
def make_scatterplot(x, y, higher_order_x, higher_order_y, snho, sho, xlabel, ylabel, assay,):
    plt.figure(figsize=(9, 6))
    plt.subplots_adjust(right=0.65)
    plt.scatter(x, y, color="blue", label="not in higher order")
    plt.scatter(higher_order_x, higher_order_y, color="orange", label='in higher order')

    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.title(f"Assay: {assay}", fontsize=18)
    
    # Add regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    plt.plot(x, intercept + slope * x, color='red')
    
    # Calculate and display R-squared value
    r_squared = r2_score(y, intercept + slope * x)
    # plt.figtext(0.02, 0.9, "bigggggggggggg", fontsize=14)
    #plt.text(0.05, 0.80, f"R-squared: {r_squared:.2f}", transform=plt.gca().transAxes, fontsize=15)
    plt.text(1.01, 1, f"no HO Spearman's: {snho:.2f}", transform=plt.gca().transAxes, 
             horizontalalignment='left', verticalalignment='top', fontsize=15)
    if len(higher_order_x) != 0:
        plt.text(1.01, 0.95, f"HO Spearman's: {sho:.2f}", transform=plt.gca().transAxes, 
                 horizontalalignment='left', verticalalignment='top', fontsize=15)
    plt.text(1.01, 0.90, f"no HO Variant Count: {len(x) - len(higher_order_x)}", transform=plt.gca().transAxes, 
             horizontalalignment='left', verticalalignment='top', fontsize=15)
    plt.text(1.01, 0.85, f"HO Variant Count: {len(higher_order_x)}", transform=plt.gca().transAxes, 
             horizontalalignment='left', verticalalignment='top', fontsize=15)
    #plt.text(0.05, 0.70, f"Unique Sites: {sho:.2f}", transform=plt.gca().transAxes, fontsize=15)
    plt.legend()
    # cbar = plt.colorbar()
    # cbar.ax.tick_params(labelsize=20)
    # cbar.ax.set_ylabel(cname, fontsize=20)
    plt.savefig(f"scatter_{xlabel}_{ylabel}_{assay}.png")
    #plt.show()
    plt.close()
    return r_squared

# do this for both r squared and spearman's
def simple_hist(values1, values2, xlabel1, xlabel2):
    plt.hist(values1, bins=10, color='blue', alpha=0.5, label=xlabel1)
    plt.hist(values2, bins=10, color='red', alpha=0.5, label=xlabel2)
    plt.ylabel('Frequency')
    plt.title('Correlation Scores and R Squared Across all Assays')
    plt.legend()
    plt.savefig(f"hist_all_assays.png")
    plt.show()
    plt.close()

# differentiate the variants that are True DMS - Sum DMS, > or < such that 90% are discarded for each assay
# this would only be for the large lineplots 
# don't write that here



# if anything, it would be nice for this to be parallelized
def eval_loop(intersect_set, desired, full, LLRS, output_csv):
    mm_full = full[full['mutant'].str.contains(":")]
    desired = ["DOCK1_MOUSE_Rocklin_2023_2M0Y.csv"]
    #["SDA_BACSU_Rocklin_2023_1PV0.csv"] #["RASK_HUMAN_Weng_2022_binding-RAF1.csv"]
    chad = []
    okay = []
    # need to get the full df
    sm_full = get_sm_LLR(full, LLRS)
    assert len(sm_full.assay.unique()) == len(intersect_set), f"{len(sm_full.assay.unique())} != {len(intersect_set)}"
    # for each assay, subset the data 
    for assay in intersect_set: # spawn a process for each assay
        if assay in desired:
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
            # need a dist from WT column

            if len(sm[sm.higher_order == True].index) > 50:
                
                sm["dist_from_WT"] = sm['mutant'].str.count(':').sum() +1
                print(sm)
                raise Error
                print(assay, len(sm.index))
                xlabel, ylabel = "DMS_score", "LLR"
                no_ho_x = sm[sm.higher_order == False].DMS_score
                no_ho_y = sm[sm.higher_order == False].LLR
                higher_order_x = sm[sm.higher_order == True].DMS_score
                higher_order_y = sm[sm.higher_order == True].LLR
                snho, _ = stats.spearmanr(no_ho_x, no_ho_y)
                sho, _ = stats.spearmanr(higher_order_x, higher_order_y)
                assay = assay.split(".")[0]
                # chad: spearman's 0.4 or greater in HO and 50% or more of total in HO
                if (sho >= 0.4) and (len(higher_order_x)/len(sm.index) > 0.5):
                    chad.append(assay)
                # okay: spearman's 0.4 or greater and at least 100 in HO
                # special okay: TCRG1_Mouse
                if (sho >= 0.4) and (len(higher_order_x) > 100):
                    okay.append(assay)
                r_squared = make_scatterplot(sm.DMS_score,sm.LLR,higher_order_x,higher_order_y,
                                            snho, sho, xlabel, ylabel, assay)
            
            # records.append({"assay": assay, "eval_size": len(sm.index), "features": "LLR", 
            #     "dist_from_WT": 1, "correlation_score":, s, "r_squared": r_squared,})
    okay += ["TCRG1_MOUSE_Rocklin_2023_2M0Y.csv"]
    return chad, okay
    #Convert the list of records into a DataFrame
    # all_records = pd.DataFrame(records)
    # make a histogram from the correlation scores in this df
    # Make a histogram from the correlation scores
    # simple_hist(all_records.correlation_score, all_records.r_squared,
    #              "Spearman's", "R_squared")
    # all_records.to_csv(output_csv) #"MM_Assay_splits.csv")



# I want another graph: one where for each feature type, we get a line plot of how the correlation changes with distance from WT
def results_lineplot(group_data, title, figname, 
                     redux=False, 
                     all_assays=False):


    plt.figure(figsize=(20, 8))
    print("all data", group_data)
    # add some logic to make it work for all assays
    if all_assays: # I have two different twos right now
        # sum the eval size for each dist from wt,
        group_data["full_eval_size"] = group_data.groupby("dist_from_WT")["eval_size"].transform("sum")


        # sum the number of assays for each dist from wt
        group_data["full_assay_size"] = group_data.groupby("dist_from_WT")["assay"].transform("nunique")
        
        # Convert the new columns to integers
        group_data['dist_from_WT'] = group_data['dist_from_WT'].astype(int)
        group_data['full_eval_size'] = group_data['full_eval_size'].astype(int)

        # Sort the DataFrame by 'dist_from_WT' and 'full_eval_size'
        group_data = group_data.sort_values(['dist_from_WT', 'full_eval_size'])
        # create the combo of dist_form_WT and eval_size
        group_data["X-axis"] = group_data.apply(lambda row: f"{row['dist_from_WT']},{row['full_eval_size']},{row['full_assay_size']}", axis=1)
        #group_data["X-axis"] = group_data.apply(lambda row: f"{row['dist_from_WT']}, {row['full_eval_size']}", axis=1)
        print('group_data["X-axis"]', group_data["X-axis"] )


    else:
        # create the combo of dist_form_WT and eval_size
        group_data["X-axis"] = group_data.apply(lambda row: f"{row['dist_from_WT']}, {row['eval_size']}", axis=1)

    
    ax = sns.lineplot(x ='X-axis', #x='dist_from_WT',
                       y='correlation_score',
                 hue = 'features', 
                 style = 'embed_used',
                 #style = 'alpha',
                 palette=color_mapping,
                 hue_order=color_order,
                 errorbar=None,
                #hue='alpha', 
                 data=group_data)
    
    plt.title(title, fontsize=30) #f'Assay: {assay}, Distance from WT: {dist_from_WT}, Evaluation Size:{eval_size}')
    if all_assays:
        plt.xlabel('Distance from WT, Number of Variants, Number of Assays', fontsize=30)
    else:
        plt.xlabel('Distance from WT, Number of Variants', fontsize=30)
    plt.ylabel('Correlation', fontsize=30)
    plt.xticks(
            #    np.arange(len(group_data['X-axis'].unique())), 
            #    group_data['X-axis'].unique(), 
               rotation=45, #'vertical',
               fontsize=12)
    plt.yticks(fontsize=12)
    legend = plt.legend(title='Features and Alpha', fontsize=15)
    legend.get_title().set_fontsize('15')  # Set the font size of the legend title
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    if all_assays:
        # Get current x-axis limits
        x_start, x_end = plt.xlim()
        print("x_start", x_start, "x_end", x_end)
        # Set new x-axis limits and x-ticks
        plt.xlim(x_start + 0.5, x_end - 0.5)  # Adjust as needed
        x_start, x_end = plt.xlim()
        print("x_start", x_start, "x_end", x_end)
        #plt.xticks(np.arange(x_start, x_end, step=0.1))  # Adjust the step as needed

    print(figname)
    plt.savefig(figname, bbox_inches='tight') #f"SM_pred_{assay}_{dist_from_WT}.png")# show()
    plt.close()

def main(args):
    output_csv = "SM_filter_Assay_splits.csv"
    if not args.graphs_only:
        # desired assays:
        if args.only_assay is None:
            #query_string =  f"{args.pg_sub_dir}/*.csv"
            query_string =  "../ESM_variant_sweep/Protein_Gym/ProteinGym_substitutions/DOCK1_MOUSE_Rocklin_2023_2M0Y.csv"
            intersect_set, full = read_in_PG(query_string)
            
            WT_dict, LLRS = get_LLR(intersect_set, full, args.llr_csv)
            chad, okay = eval_loop(intersect_set, intersect_set, full, LLRS, output_csv)
            # now get the lineplots for each assay and the mean plots for each group

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
