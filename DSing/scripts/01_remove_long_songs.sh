# /bin/bash
#
# Find and remove long songs.  Run this in the directory where you original ran wav2vec_create_split_labels.py.
# Note: you should run this before you create the metadata.csv files...
#

logfile="removed_files.log"

echo "Removing the following files" | tee -a $logfile
for file in `ls *.wav`; do
    duration=`ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 $file;`
    if (( $(echo "$duration > 28.0" | bc -l) )); then
        echo "$file" | tee -a $logfile
        rm $file
    fi
done
