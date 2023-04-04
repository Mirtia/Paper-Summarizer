#!/bin/bash

file="YouTubers_Not_madeForKids_Detecting_Channels_Sharing_Inappropriate_Videos_Targeting_Children"
mkdir -p data/output
echo -e "Testing summarization methods ...\n"
echo -e "\nSummarization nltk:\n"
time python src/main.py -f data/input/${file}.pdf -o data/output/${file}_nltk.txt -m nltk
# time python src/main.py -f data/input/${file}.pdf -o data/output/${file}_pegasus.txt -m pegasus
echo -e "\nSummarization sumy:\n"
time python src/main.py -f data/input/${file}.pdf -o data/output/${file}_summy.txt -m sumy