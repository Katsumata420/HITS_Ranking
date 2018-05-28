import collections
import utilities as util
import sklearn.feature_extraction
import numpy as np

def extract_context(input_file, window_size):
    """
    this function is counting the co-occurrence in the words at the input file.
    co_occur: one word to co-occurrence word
    word_occur: occur word (as token)
    context_occur: co-occurrence word
    This function is a very naive implementation; I want to devise a little more.
    """

    co_occur = collections.defaultdict(int)
    word_occur = collections.defaultdict(int)
    context_occur = collections.defaultdict(int)

    i_f = open(input_file)
    
    for num, line in enumerate(i_f):
        if num % 10000 == 0:
            util.trace('look at sent: '+str(num))
        words = line.strip().split()

        for i in range(len(words)):
            # left word  
            for j in range(1, window_size+1):
                if i - j > -1:
                    co_occur['{}\t{}'.format(words[i], words[i-j])] += 1
                    word_occur[words[i]] += 1
                    context_occur[words[i-j]] += 1
                else:
                    continue
            
            #right word
            for j in range(1, window_size+1):
                if i + j < len(words):
                    co_occur['{}\t{}'.format(words[i], words[i+j])] += 1
                    word_occur[words[i]] += 1
                    context_occur[words[i+j]] += 1
                else:
                    continue
    i_f.close()
    
    return co_occur, word_occur, context_occur

def make_matrix(co_occur, word_occur, context_occur, model, th=1):
    """
    this function is making the matrix expresses word-graph.
    word_name: word to id
    num_pairs: sum of the all co-occurrence
    feature_list: a list the elemint of which is dict (key; co-occurrence, value: occurrence or pmi).
    co_word_dict: key is a word, values are the co-occurrence words around the word.
    word_dict: key is the co-occurrence word, value is occurrence (Freq) or PPMI value (PPMI).
    """
    vectorizer = sklearn.feature_extraction.DictVectorizer()
    word_name = dict()
    feature_list = list()
    co_word_dict = collections.defaultdict(list)
    num_pairs = 0

    for value in co_occur.values():
        num_pairs += value

    for word in co_occur.keys():
        w, c = word.split('\t')

        # if only once appearance (or under threshold) of the co-occurrence
        # the matrix element is 0.
        if co_occur[word] <= th:
            c = c+'@@@@@None'
        co_word_dict[w].append(c)

    counter = 0

    # caluculation of the matrix element
    for word, co_words in sorted(co_word_dict.items()):
        if counter % 10000 == 0:
            util.trace('add word num :'+str(counter))
        
        word_name[word] = counter
        word_dict = dict()

        for c in co_words:
            if c.find('@@@@@None') != -1:
                word_dict[c.rsplit('@@@@@None')[0]] = 0
            else:
                if model == 'Freq':
                    word_dict[c] = co_occur['{}\t{}'.format(word, c)]
                elif model == 'PPMI':
                    pmi_value = returnPmi(word, c, co_occur, word_occur, context_occur, num_pairs)
                    pmi_value += np.log2(co_occur['{}\t{}'.format(word, c)])
                    
                    word_dict[c] = max(0, pmi_value)
        
        feature_list.append(word_dict)
        counter += 1
    
    matrix = (word_name, vectorizer.fit_transform(feature_list))

    return matrix, vectorizer

def returnPmi(word, context, co_occur, word_occur, context_occur, num_pairs):
    pmi = np.log2(num_pairs*co_occur['{}\t{}'.format(word, context)]/word_occur[word]/context_occur[context])
    return pmi



