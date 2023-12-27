# Python standard libraries
import argparse

# Third-party libraries
import pandas as pd
import numpy as np
import torch
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer


def detect_max_batch_size(model, fasta, alphabet, device_id, truncation_seq_length):
    dataset = FastaBatchedDataset.from_file(fasta)
    # push the model to device
    model = model.to(device_id)
    model.eval()
    # start big and go down, with a binary search.
    toks_per_batch = 1000000
    forward = False
    # 1 mil -> 500,000 -> 250,000 -> 125,000 -> 62,500 -> 31,250 -> 15,625 -> 7,812 -> 3,906. Should stop here for most gpus
    while not forward:
        batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
        data_loader = torch.utils.data.DataLoader(
            dataset, collate_fn=alphabet.get_batch_converter(truncation_seq_length), batch_sampler=batches
        )
        with torch.no_grad():
            # this dataloader will give me the smallest strings first. I want the biggest to cause OOM ASAP
            forward_arr = []
            for batch_idx, (labels, strs, toks) in enumerate(data_loader):
                if batch_idx >= len(batches) - 5:
                    # This is one of the last 5 batches
                    if torch.cuda.is_available() and not args.nogpu:
                        toks = toks.to(device=f"cuda:{device_id}", non_blocking=True)
                    try: # attempt the forward pass
                        out = model(toks, repr_layers=[33], return_contacts=False)
                        forward_arr.append(True)
                    except RuntimeError as e:
                        forward_arr.append(False)
    if all(forward_arr) == True:
        forward = True
    else:
        toks_per_batch /= 2
    print(f"Maximum batch size for this model is {toks_per_batch}")
    print(f"Read {fasta} with {len(dataset)} sequences")
    return model, data_loader, batches
        


def get_model(model_name, fasta, device_id): 
    truncation_seq_length = 1022 # this is the max length of the sequence, longer will be truncated prior to forward pass
    model, alphabet = pretrained.load_model_and_alphabet(model_name)
    # put model on device and detect max batch size
    model, data_loader, batches = detect_max_batch_size(model, fasta, alphabet, device_id, truncation_seq_length)
    # use torch compile to maximize performance
    model = torch.compile(model, dynamic=True, mode="max-autotune" )
    return model, alphabet, data_loader, batches

# right now this only works for multi missense, not indels
def get_PLLR(model, alphabet, data_loader, batches, device_id):
    # let's remake the df mut_seq, esm_score

    all_PLLRs, all_strs = [], []
    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(
                f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
            )
            # gotta mod this
            if torch.cuda.is_available() and not args.nogpu:
                toks = toks.to(device=f"cuda:{device_id}", non_blocking=True)
            # get the logits
            out = model(toks, repr_layers=[33], return_contacts=False)
            logits = out["logits"]
            s = torch.log_softmax(logits,dim=-1).cpu().numpy()
            s = s[0][1:-1,:]
            # so now we need all the seqs for the batch
            PLLRs = np.zeros(len(strs))
            for j in range(len(strs)): #this worked
                seq = strs[j]
                idx=[alphabet.tok_to_idx[i] for i in seq]
                PLLR = np.sum(np.diag(s[:,idx]))
                PLLRs[j] = PLLR
            # now you have all PLLRs for this batch, collect them
            all_PLLRs.append(PLLRs)
            all_strs += strs

    all_PLLRs, all_strs = np.vstack(all_PLLRs), np.array(all_strs)
    # Create the DataFrame
    df = pd.DataFrame({'mut_seq': all_strs, 'esm_score': all_PLLRs,})
    return df
    

def main(args):
    """
    Execute the main script logic.
    
    Parameters:
    args (Namespace): Arguments parsed from command line input.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {}.'.format('GPU' if device == 'cuda' else 'CPU (this may be much slower)'))

    # read the dataframe 
    input_df = pd.read_csv(args.input_csv_file)
    # detect multiple GPUs
    if device == 'cuda': # this will override cuda and now be [0, 1,2, 3 ...]
        gpu_ids = list(range(torch.cuda.device_count()))
        print('Available GPU IDs:', gpu_ids)
        # Shuffle the dataframe rows
        input_df = input_df.sample(frac=1, random_state=42)

        # Sort the dataframe by the length of the string column in ascending order
        input_df['string_length'] = input_df['string_column'].apply(len)
        input_df = input_df.sort_values('string_length')

        # Shuffle the dataframe rows again to distribute strings of all lengths
        input_df = input_df.sample(frac=1, random_state=42)

        # Remove the temporary 'string_length' column
        input_df = input_df.drop('string_length', axis=1)
        
        # Split the dataframe into smaller dataframes
        input_dfs = np.array_split(input_df, len(gpu_ids))

    print('Loading the model ({})...'.format(args.model_name))
    # need to keep track of an array of models, they now exist on different gpus
    output_df_ls = []
    if device == 'cuda':
        for i in len(gpu_ids):
            model, alphabet, data_loader, batches = get_model(args.model_name, input_dfs[i], gpu_ids[i])
            output_df_ls.append(get_PLLR(model, alphabet, data_loader, batches, gpu_ids[i]))
        # then we need to concatenate the output_df_ls
        output_df = pd.concat(output_df_ls)
        print('Saving results...')
        output_df.to_csv(args.output_csv_file, index=False)
        print('Done.')
            
    else:
        model, alphabet, data_loader, batches = get_model(args.model_name, input_df, device)
        output_df = get_PLLR(model, alphabet, data_loader, batches, device)
        print('Saving results...')
        output_df.to_csv(args.output_csv_file, index=False)
        print('Done.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute ESM effect scores for specified multi-residue variants in a set of protein sequences.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--input-fasta-file', dest='input_fasta_file', required=True, metavar='/path/to/input_mutations.fasta', help='Path to the input CSV file with the protein mutations to calculate ESM scores for.')
    parser.add_argument('--output-csv-file', dest='output_csv_file', required=True, metavar='./esm_multi_residue_effect_scores.csv', help='Path to save the output CSV file.')
    parser.add_argument('--model-name', dest='model_name', default='esm1b_t33_650M_UR50S', metavar='esm1b_t33_650M_UR50S', help='Name of the ESM model to use. See list of options here: https://github.com/facebookresearch/esm#available')

    args = parser.parse_args()

    main(args)