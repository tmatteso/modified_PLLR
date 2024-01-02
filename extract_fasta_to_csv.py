import pandas as pd
import argparse

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
                        print(name)
                        print(sequence)
                        raise Error
                    name = line[1:]
                    sequence = ''
                else:
                    sequence += line
            if name is not None:
                names.append(name)
                sequences.append(sequence)
        df = pd.DataFrame({'name': names, 'sequence': sequences})
        return df 
def main(args):
    # read in the fasta file as a pandas df
    # Read the fasta file
    records = read_fasta_file(args.input_fasta_file)

    # Convert the records to a pandas DataFrame
    df = pd.DataFrame(records)

    # convert it to a df with 3 rows: wt_seq,mut_seq,start_pos
    df = pd.DataFrame({'wt_seq': df['sequence'], 'mut_seq': df['sequence'], 'start_pos': 0})

    # output the csv
    df.to_csv(args.output_csv_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='covert bulk fasta extraction format to nadav script format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--input-fasta-file', dest='input_fasta_file', required=True, metavar='/path/to/input_mutations.fasta', help='Path to the input CSV file with the protein mutations to calculate ESM scores for.')
    parser.add_argument('--output-csv-file', dest='output_csv_file', required=True, metavar='./esm_multi_residue_effect_scores.csv', help='Path to save the output CSV file.')
    args = parser.parse_args()

    main(args)