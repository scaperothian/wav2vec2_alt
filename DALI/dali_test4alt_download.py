"""
Contains methods for reading in the csv for the DALI Test split. 

Originates from E. Demiril and is a subset of DALI v1.0.
CSV file can alternatively be found at: https://github.com/emirdemirel/DALI-TestSet4ALT
"""
import os
import pandas as pd

from dali_downloader import get_song_by_songid

import logging
logging.basicConfig(format='%(message)s',
                            filename='dali_test4alt_download.log',
                            filemode='w',
                            level=logging.DEBUG)

if __name__ == "__main__":

    logging.debug("******* RUNNING DALI TEST4ALT DOWNLOADER ************")
    
    # TODO: change this to argparse
    INPUT_CSV_DIR = './DALI_TestSet4ALT.csv'
    SAVE_DIR = './audio_test'

    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    data = pd.read_csv(INPUT_CSV_DIR,index_col=0)
    nrows,ncols = data.shape

    # Note: my convention for downloading the data was to: 
    # (1) clone this repo
    # (2) download the 'data' from Zenodo (i.e. the metdata) and save under metadata_v{1,2}/
    # (3) download audio_v{1,2} for the full corpus
    # (4) run this script to pull data into audio_test/
    dali_data_path = os.getcwd()+"/metadata_v1"
    audio_path = os.getcwd()+"/audio_test"

    for i, row in enumerate(data.iterrows()):
        logging.debug(f"************** SONG {i} / {nrows} **************")

        songid = row[0]
        get_song_by_songid(dali_data_path, SAVE_DIR, songid)
        