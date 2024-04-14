import yt_dlp
import sys
import os
import numpy as np
import time
import argparse
import time

import logging
logging.basicConfig(format='%(message)s',
                            filename='dali_downloader.log',
                            filemode='w',
                            level=logging.DEBUG)

import DALI as dali_code
from DALI import utilities

base_url = 'http://www.youtube.com/watch?v='

class MyLogger(object):
    def debug(self, msg):
        print(msg)

    def warning(self, msg):
        print(msg)

    def error(self, msg):
        print(msg)


def my_hook(d):
    if d['status'] == 'finished':
        print('Done downloading, now converting ...')
        
def get_ydl(outtmpl):
   
    ydl_opts = {'format': 'bestaudio/best',
                        'postprocessors': [{'key': 'FFmpegExtractAudio',
                                            'preferredcodec': 'wav',
                                            'preferredquality': '320'}],
                        'outtmpl': outtmpl,
                        'logger': MyLogger(),
                        'progress_hooks': [my_hook],
                        'verbose': False,
                        'ignoreerrors': False,
                        'external_downloader': 'ffmpeg',
                        'nocheckcertificate': True}
                        # 'external_downloader_args': "-j 8 -s 8 -x 8 -k 5M"}
                        # 'maxBuffer': 'Infinity'}
                        #  it uses multiple connections for speed up the downloading
                        #  'external-downloader': 'ffmpeg'}
    ydl = None
    try: 
        ydl = yt_dlp.YoutubeDL(ydl_opts)
    except Exception as e:
        print(f"get_ydl: {e}")
    return ydl

def get_song_by_songid(dali_data_path, audio_path, songid):
    """
    Download a single file and place it in the audio_path.  Fetch audio metadata from dali_data_path.
    
    audio_path (string) - absolute directory path. 
    dali_data_path (string) - absolute directory path.  
    youtube_url (string) - string for the youtube path.

    """
    filename = audio_path+"/"+songid+".wav"
    dali_data = dali_code.get_the_DALI_dataset(dali_data_path, skip=[], keep=[songid])
    entry = dali_data[songid]
    logging.debug(entry.info)

    youtube_url = entry.info['audio']['url']

    # Check if the file exists
    if os.path.isfile(filename):
        logging.debug(f"{os.path.split(filename)[-1]} exists already.  Moving on.")
    else:
        print(f"Song: {songid}.wav from {youtube_url}")    
        ydl = get_ydl(audio_path+"/"+songid)
        if ydl:
            try: 
                ydl.download([base_url + youtube_url])
            except Exception as e:
                logging.debug(f"ERROR: {e}")
                f = open(audio_path+"/"+songid+".log","w")
                f.write(youtube_url+"\n\n"+str(e)+"\n")
                f.close()
        else:
            print(f"ISSUE DOWNLOADING: {url}")

def get_song_by_index(dali_data_path, audio_path, n):
    """
    Download a single file and place it in the audio_path.  Fetch audio metadata from dali_data_path.
    
    audio_path (string) - absolute directory path. 
    dali_data_path (string) - absolute directory path.  
    n (int) - DALI index used to reference song. 
    """
    dali_info = dali_code.get_info(dali_data_path +'/'+ 'info/DALI_DATA_INFO.gz')
    allsongfilenames = utilities.get_files_path(dali_data_path,'.gz')

    print(dali_info[n])
    
    songid = os.path.relpath(allsongfilenames[n],dali_data_path).split('.')[0]
    
    # for testing...
    dali_data = dali_code.get_the_DALI_dataset(dali_data_path, skip=[], keep=[songid])
    entry = dali_data[songid]
    print(entry.info)
    
    filename = audio_path+"/"+songid+".wav"
    # Check if the file exists
    if os.path.isfile(filename):
        print(f"{os.path.split(filename)[-1]} exists already.  Moving on.")
    else:
        url = dali_info[n][2]
        print(f"Song {n}: {songid}.wav from {url}")    
        ydl = get_ydl(audio_path+"/"+songid)
        if ydl:
            try: 
                ydl.download([base_url + url])
            except Exception as e:
                print(f"ERROR: {e}")
                f = open(audio_path+"/"+songid+".log","w")
                f.write(url+"\n\n"+str(e)+"\n")
                f.close()
        else:
            print(f"ISSUE DOWNLOADING: {url}")

def get_five_random_songs(dali_data_path, audio_path):

    dali_data = dali_code.get_the_DALI_dataset(dali_data_path, skip=[], keep=[])
    all_songids = list(dali_data.keys())
    
    # get random 5 songs
    for _ in range(5):
        n = np.random.randint(0,len(all_songids))
        songid = all_songids[n]
        entry = dali_data[songid]

        youtube_url = entry.info['audio']['url']
        
        print(f"Song {n}: {songid}.wav from {youtube_url}")
        ydl = get_ydl(audio_path+"/"+songid)
        if ydl:
            try: 
                ydl.download([base_url + youtube_url])
            except Exception as e:
                print(f"ERROR: {e}")
                f = open(audio_path+"/"+songid+".log","w")
                f.write(youtube_url+"\n\n"+str(e)+"\n")
                f.close()
        else:
            print(f"ISSUE DOWNLOADING: {youtube_url}")

def generic_download(youtube_url,filename):
    ydl = get_ydl(filename)
    if ydl:
        try: 
            ydl.download([base_url + youtube_url])
        except Exception as e:
            print(f"ERROR: {e}")
    else:
        print(f"ISSUE DOWNLOADING: {youtube_url}")

def get_all_songs(dali_data_path, audio_path):
    
    dali_data = dali_code.get_the_DALI_dataset(dali_data_path, skip=[], keep=[])
    all_songids = list(dali_data.keys())
    nsongs = len(all_songids)
    
    looptime = 0
    for i, songid in enumerate(all_songids):
        entry = dali_data[songid]
        youtube_url = entry.info['audio']['url']
        
        print(f"************** SONG {i} / {nsongs} **************")
        start = time.time()

        filename = audio_path+"/"+songid+".wav"
        
        # Check if the file exists.  Don't waste bandwidth if already downloaded...
        # This is nice if i've tried a few times and gotten rate limited...
        #
        if os.path.isfile(filename):
            print(f"{filename} exists already.  Moving on.")
        else:
            print(f"Song {i}: {songid}.wav from {youtube_url}")
            ydl = get_ydl(audio_path+"/"+songid)
            if ydl:
                try: 
                    ydl.download([base_url + youtube_url])
                except Exception as e:
                    #print(f"ERROR: {e}")
                    logname = audio_path+"/"+songid+".log"
                    with open(logname,"w") as file:
                        file.write(youtube_url+"\n\n"+str(e)+"\n")
    
            else:
                print(f"ISSUE DOWNLOADING: {url}")
            end = time.time()
            looptime = end-start
            nextsongfile = ".nextsong"
            with open(nextsongfile,"w") as file:
                file.write(f"{i+1}\n")
    
            looptimefile = ".looptime"
            with open(looptimefile,"w") as file:
                file.write(f"{looptime}\n")
    
            time.sleep(1)


if __name__ == "__main__":
    #Example Usage: 
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_path", type=str, default="metadata", help="Absolute Parent directory of DALI audio metadata (downloaded from Zenodo).")
    parser.add_argument("--audio_path", type=str, default="audio", help="Absolute Directory for audio data downloaded from Youtube.")

    args = parser.parse_args()
    get_all_songs(args.metadata_path, args.audio_path)


