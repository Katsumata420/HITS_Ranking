# HITS_Ranking
This is a tool code of ranking words using HITS algolithm, described in the following paper:  
- Satoru Katsumata, Yukio Matsumura, Hayahide Yamagishi and Mamoru Komachi. 2018. 
Graph-based Filtering of Out-of-Vocabulary Words for Encoder-Decoder Models. In Proc. of ACL 2018, Student Research Workshop.  

## 1. Environmental Settings  
You have to install these modules. (version)  
- Python3 (3.6.0)
- numpy (1.13.3)
- scikit-learn (0.19.1)
- scipy (1.0.0)

## 2. Experimental Settings  
- **inputF**: input file (one line one sentence)
- **model**: Freq or PPMI
- **outF**: output folder (the ranked words, word-graph matrix, and vectorizer are saved.)
- **window**: the window size of defining co-occurrence
- **iter**: HITS iteration (The larger the words in the input text are, the bigger the number of it should be.)
- **vocabSize**: vocabulary size of output

If you try this script, please `sh quick_start.sh`. 
