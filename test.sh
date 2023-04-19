#!/bin/bash

title=$1
mkdir -p data/output
echo -e "Testing summarization methods ...\n"
echo -e "\nSummarization nltk:\n"
time python src/main.py -f data/input/${title}.pdf -o data/output/${title}_nltk.txt -m nltk
echo -e "\nSummarization sumy:\n"
time python src/main.py -f data/input/${title}.pdf -o data/output/${title}_summy.txt -m sumy
# echo -e "\nSummarization pegasus:\n"
# time python src/main.py -f data/input/${title}.pdf -o data/output/${title}_pegasus.txt -m pegasus