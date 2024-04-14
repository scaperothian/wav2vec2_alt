"""
Creating metadata.csv file required for a dataset split (i.e. test/dev/train)
"""
import os
import glob
import pandas as pd

def find_wav_files(path):
    """
    Find all .wav files in the specified directory and its subdirectories.
    
    Args:
    - directory (str): The directory path to search for .wav files.
    
    Returns:
    - List of strings: Paths to all .wav files found.
    """
    # Define the pattern to search for .wav files
    pattern = os.path.join(path, '*')
    # Use glob to find all files matching the pattern
    wav_files = glob.glob(pattern, recursive=False)
    
    return [os.path.basename(path) for path in wav_files]



def read_labels_file(labels_path):
    """
    read Kaldi generated text file, return dataframe with two columns
    """
    columns = ['ID', 'Text']
    # Read the text file and add a comma after the first element in each line
    with open(labels_path, 'r') as file:
        lines = [line.strip().split(maxsplit=1) for line in file]
        lines = [f"{parts[0]}, {parts[1]}" for parts in lines]
    
    return pd.DataFrame([line.split(', ', 1) for line in lines], columns=columns)

if __name__ == "__main__":
    
    # Reminder to future self.  The first step to creating this dataset was to run a 
    # Kaldi recipe from https://github.com/groadabike/Kaldi-Dsing-task
    # This creates the wav file chunks and a form of the <split>/text file
    # that gives you what you need for creating the metadata file.
    
    #TODO: create parseargs
    split='TEST'
    
    # read in a file from DSing
    dataset_folder = os.getcwd()+"/DALI_Test"


    if split=='TEST':
        # getting things ready for the test split
        utt_folder = dataset_folder
        utt_labels_file  = dataset_folder
    else:
        print('Bad split selected.')
        exit()
    
    # get lables
    df_labels = read_labels_file(utt_labels_file)
    
    # adding the actual file name to the ID column
    df_labels['ID'] = df_labels['ID'] + ".wav"

    # Rename the columns to fit the pytorch conventions.
    df_labels = df_labels.rename(columns={'ID': 'file_name', 'Text': 'transcription'})
    
    destination_dir = utt_folder+"/metadata.csv"
    df_labels.to_csv(destination_dir,index=False)
    print(f"Creating metadata.csv in {destination_dir}") 
    