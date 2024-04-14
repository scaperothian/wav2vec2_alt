#
# This should be something that i never use again...
# I *should* remove long files with remove_long_songs.sh first
# then i should run wav2vec_create_split_metadata.py
#
# i did this in the opposite order and created this script to fix the metadata file.
#
if __name__=="__main__":
    SPLIT='train1'
    
    # read in the original metadata file created with wav2vec_create_split_labels.py
    dsing_train_metadata = pd.read_csv(f'../sing_300x30x2/damp_dataset/{SPLIT}/metadata.csv')
    print(dsing_train_metadata.shape)
    print(dsing_train_metadata.columns)

    # read in the log file from the remove_long_song.sh script
    # this tells you exactly which songs to pull out.
    removed_files = pd.read_csv(f'../sing_300x30x2/damp_dataset/{SPLIT}/removed_files.log')
    
    j = 0
    for i, ds_removed_file in removed_files.iterrows():
        file = ds_removed_file['file_name']
        if file in dsing_train_metadata['file_name'].values:
                dsing_train_metadata = dsing_train_metadata[file!=dsing_train_metadata['file_name']]

    # saves to the local file
    file_to_save = "metadata.csv"
    dsing_train_metadata.to_csv(file_to_save,index=False)
    print(f"Creating {file_to_save}") 