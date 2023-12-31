# Python standard libraries
import argparse

# Third-party libraries
import pandas as pd
import numpy as np
import torch
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
import multiprocessing
import pathlib
import time

def detect_max_batch_size(model, fasta, alphabet, device_id, truncation_seq_length):
    # would be nice to have a device map for quick reference: torch.cuda.get_device_name(device_id)
    # build some simple map st T4 -> 62500, V100 -> 125000, etc.

    # push the model to device
    model = model.to(device_id)
    model.eval()
    #print(model)
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))
    # print("oh hi yo")
    # time.sleep(20)
    model.eval()
    dataset = FastaBatchedDataset.from_file(fasta)
    # start big and go down, with a binary search.
    toks_per_batch = 31250 #62500 # this seems optimal for T4 GPUs
    forward = False
    # 1 mil -> 500,000 -> 250,000 -> 125,000 -> 62,500 -> 31,250 -> 15,625 -> 7,812 -> 3,906. Should stop here for most gpus
    print(f"Attempting to find maximum batch size for model on device {device_id}")
    while not forward:
        print(f"Trying batch size: {toks_per_batch} on {device_id}")
        batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
        data_loader = torch.utils.data.DataLoader(
            dataset, collate_fn=alphabet.get_batch_converter(truncation_seq_length), batch_sampler=batches
        )
        with torch.no_grad():
            # this dataloader will give me the smallest strings first. I want the biggest to cause OOM ASAP
            forward_arr = []
            for batch_idx, (labels, strs, toks) in enumerate(data_loader):
                # print("oh hi yo")
                # time.sleep(20)
                if batch_idx >= len(batches) - 5:
                    # This is one of the last 5 batches
                    if torch.cuda.is_available():
                        toks = toks.to(device=f"cuda:{device_id}", non_blocking=True)
                    # print(toks.shape)
                    try: # attempt the forward pass
                        # time.sleep(20)
                        out = model(toks, repr_layers=[33], return_contacts=False)
                        
                        forward_arr.append(True)
                    except RuntimeError as e:
                        print(f"Batch {batch_idx} failed on device {device_id}")
                        forward_arr.append(False)
                        break
        if all(forward_arr) == True:
            forward = True
        else:
            toks_per_batch = toks_per_batch // 2
    print(f"Maximum batch size for model on device {device_id} is {toks_per_batch}")
    print(f"Read {fasta} with {len(dataset)} sequences")
    return model, data_loader, batches
        


def get_model(model_name, fasta, device_id): 
    truncation_seq_length = 1022 # this is the max length of the sequence, longer will be truncated prior to forward pass
    model, alphabet = pretrained.load_model_and_alphabet(model_name)
    # put model on device and detect max batch size
    model, data_loader, batches = detect_max_batch_size(model, fasta, alphabet, device_id, truncation_seq_length)
    # use torch compile to maximize performance -- right now this causes a bug with multiprocessing, it spawns child processes
    # model = torch.compile(model, dynamic=True, mode="max-autotune" )
    return model, alphabet, data_loader, batches

# right now this only works for multi missense, not indels
def get_PLLR(model, alphabet, data_loader, batches, device_id, args):
    # let's remake the df mut_seq, esm_score
    all_PLLRs, all_strs = [], []
    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(
                f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences) on device {device_id}"
            )
            # gotta mod this
            if torch.cuda.is_available():
                toks = toks.to(device=f"cuda:{device_id}", non_blocking=True)
            # get the logits
            out = model(toks, repr_layers=[21, 33], return_contacts=False)
            logits = out["logits"] # this first one is fine. it fails after. We observed this behavior before
            #logits = logits[0]
            #logits = logits.unsqueeze(0)
            s = torch.log_softmax(logits,dim=-1).cpu().numpy()
            #s = s[0][1:-1,:] # this is definitely wrong. Just based on the shapes. You are only choosing the logit output of the first input.
            # the way you compute logits only works for the first sequence in the batch and is wrong for all other sequences
            # the dataloader, sting i/o, and alphabet are blameless.

            # For now, I want it to collect the representations too
            representations = {
                layer: t.to(device="cpu") for layer, t in out["representations"].items()
            }

            # so now we need all the seqs for the batch
            PLLRs = np.zeros(len(strs))
            # we need to redo the validation

            for j in range(len(strs)): #this worked
                #[number of sequences, length of sequences, number of layers]
                # logits = out["logits"] # this first one is fine. it fails after. We observed this behavior before
                # logits = logits[0]
                # logits = logits.unsqueeze(0)
                # s = torch.log_softmax(logits,dim=-1).cpu().numpy()
                #print(s.shape)
                s_j = s[j][1:-1,:]
                seq = strs[j]
                idx=[alphabet.tok_to_idx[i] for i in seq]
                PLLR = np.sum(np.diag(s_j[:,idx]))

                PLLRs[j] = PLLR
                args.output_file = args.output_dir / f"{labels[j]}.pt"
                args.output_file.parent.mkdir(parents=True, exist_ok=True)
                result = {"label": labels[j]}
                truncate_len = min(1022, len(strs[j]))
                result["PLLRs"] = PLLRs[j]
                # Call clone on tensors to ensure tensors are not views into a larger representation
                # See https://github.com/pytorch/pytorch/issues/1995
                result["mean_representations"] = {
                            layer: t[j, 1 : truncate_len + 1].mean(0).clone()
                            for layer, t in representations.items()
                }
                torch.save(
                    result,
                    args.output_file,
                )

            # for i, label in enumerate(labels):
            #     args.output_file = args.output_dir / f"{label}.pt"
            #     args.output_file.parent.mkdir(parents=True, exist_ok=True)
            #     result = {"label": label}
            #     truncate_len = min(1022, len(strs[i]))
            #     result["PLLRs"] = PLLRs[i]
            #     # Call clone on tensors to ensure tensors are not views into a larger representation
            #     # See https://github.com/pytorch/pytorch/issues/1995
            #     result["mean_representations"] = {
            #                 layer: t[i, 1 : truncate_len + 1].mean(0).clone()
            #                 for layer, t in representations.items()
            #     }
            #     torch.save(
            #         result,
            #         args.output_file,
            #     )
            del out # dump these out of gpu memory
            # now you have all PLLRs for this batch, collect them
            all_PLLRs.append(PLLRs)
            all_strs += strs

    all_PLLRs, all_strs = np.concatenate(all_PLLRs), np.array(all_strs)
    # print(all_PLLRs.shape)
    #print(all_strs[:3])
    # Create the DataFrame
    df = pd.DataFrame({'mut_seq': all_strs, 'esm_score': all_PLLRs,})
    return df

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
        df = pd.DataFrame({'name': names, 'sequence': sequences})
        return df 

def worker_function(model_name, fasta, device, args):
    try:
        model, alphabet, data_loader, batches = get_model(model_name, fasta, device)
        output_df = get_PLLR(model, alphabet, data_loader, batches, device, args) 
        return output_df
    except Exception as e:
        print(f"Error in worker function: {e}")
        raise 

def main(args):
    """
    Execute the main script logic.
    
    Parameters:
    args (Namespace): Arguments parsed from command line input.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {}.'.format('GPU' if device == 'cuda' else 'CPU (this may be much slower)'))

    # detect multiple GPUs
    if device == 'cuda': # this will override cuda and now be [0, 1,2, 3 ...]
        gpu_ids = list(range(torch.cuda.device_count()))
        print('Available GPU IDs:', gpu_ids)
        input_df = read_fasta_file(args.input_fasta_file)
        # Shuffle the dataframe rows
        input_df = input_df.sample(frac=1, random_state=42)

        # Sort the dataframe by the length of the string column in ascending order
        input_df['string_length'] = input_df['sequence'].apply(len)
        input_df = input_df.sort_values('string_length')

        # Shuffle the dataframe rows again to distribute strings of all lengths
        input_df = input_df.sample(frac=1, random_state=42)

        # Remove the temporary 'string_length' column
        input_df = input_df.drop('string_length', axis=1)

        # Split the dataframe into smaller dataframes
        input_dfs = np.array_split(input_df, len(gpu_ids))

        # Create a list to store the new fasta file names
        fasta_file_names = []

        for i, df in enumerate(input_dfs):
            # Convert the DataFrame to fasta format
            fasta_data = '>' + df['name'] + '\n' + df['sequence'] + '\n'

            # Generate a new fasta file name
            orig_name  = args.input_fasta_file.split(".")[0]
            fasta_file_name = f'{orig_name}_{i}.fasta'

            # Save the fasta data to a file
            with open(fasta_file_name, 'w') as file:
                file.write(''.join(fasta_data))

            # Append the fasta file name to the list
            fasta_file_names.append(fasta_file_name)
        input_dfs = fasta_file_names

    print('Loading the model ({})...'.format(args.model_name))
    # need to keep track of an array of models, they now exist on different gpus
    if device == 'cuda' and __name__ == '__main__':
        # this code must be inside the __name__ == '__main__' to protect it from running unintentionally during the import phase        
        def parallel_processing(worker_function, process_args):
            with multiprocessing.Pool(len(process_args)) as pool:
                # each process needs different inputs
                results = pool.starmap(worker_function, process_args)
            return results
        
        multiprocessing.set_start_method('spawn')
        process_args = [ (args.model_name, f'{orig_name}_{i}.fasta', i, args) for i in range(len(gpu_ids)) ]
        results = parallel_processing(worker_function, process_args)
        # output the strings with their PLLRs for comparison
        output_df = pd.concat(results, ignore_index=True)
        print('Saving results...')
        output_name = args.input_fasta_file.split("/")[-1].split(".")[0]
        output_df.to_csv(f"{output_name}_output.csv", index=False)
        print('Done.')


        # for i in range(len(gpu_ids)):
        #     model, alphabet, data_loader, batches = get_model(args.model_name, input_dfs[i], gpu_ids[i])
        #     output_df = get_PLLR(model, alphabet, data_loader, batches, gpu_ids[i])
        #     # wait for all processes to complete
        #     output_df_ls.append(output_df)
        # # then we need to concatenate the output_df_ls
        # output_df = pd.concat(output_df_ls)
        # print('Saving results...')
        # output_df.to_csv(args.output_csv_file, index=False)
        # print('Done.')
            
    else:
        model, alphabet, data_loader, batches = get_model(args.model_name, args.input_fasta_file, device)
        output_df = get_PLLR(model, alphabet, data_loader, batches, device, args)
        print('Saving results...')
        output_name = args.input_fasta_file.split("/")[-1].split(".")[0]
        output_df.to_csv(f"{output_name}_output.csv", index=False)
        print('Done.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute ESM effect scores for specified multi-residue variants in a set of protein sequences.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--input-fasta-file', dest='input_fasta_file', required=True, metavar='/path/to/input_mutations.fasta', help='Path to the input CSV file with the protein mutations to calculate ESM scores for.')
    #parser.add_argument('--output-csv-file', dest='output_csv_file', required=True, metavar='./esm_multi_residue_effect_scores.csv', help='Path to save the output CSV file.')
    parser.add_argument('--output_dir', type=pathlib.Path, help="output directory for extracted representations",) 
    parser.add_argument('--model-name', dest='model_name', default='esm1b_t33_650M_UR50S', metavar='esm1b_t33_650M_UR50S', help='Name of the ESM model to use. See list of options here: https://github.com/facebookresearch/esm#available')

    args = parser.parse_args()

    main(args)