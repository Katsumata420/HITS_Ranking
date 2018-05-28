mkdir -p save_folder/sample
python rank_words.py data/sample.txt -model PPMI -outF save_folder/sample -window 2 -iter 300
