import argparse
import utilities as util
import word_graph
import hits
import math

def main():
    parser = argparse.ArgumentParser(usage='sorry, look at readme...', \
            description='arg description', epilog='end')
    parser.add_argument('inputF', help='write the file name of the input text.')
    parser.add_argument('-model', help='select Freq or PPMI.', default='PPMI', choices=['Freq', 'PPMI'])
    parser.add_argument('-outF', help='write the output file name.', default='sample')
    parser.add_argument('-window', help='define the window size.', type=int, default=2)
    parser.add_argument('-iter', help='the number of HITS iteration.', type=int, default=300)
    parser.add_argument('-vocabSize', help='define the vocabulary size. default is all.', type=int, default=None)
    args = parser.parse_args()

    # counting co-occurrence
    util.trace('count the co-occurrence')
    co_occur, word_occur, context_occur = word_graph.extract_context(args.inputF, args.window) 

    util.trace('vocabulary size of the input data is {}.'.format(len(word_occur)))
    if args.vocabSize:
        vocabSize = args.vocabSize
    else:
        vocabSize = len(word_occur)

    # calculate matrix
    util.trace('make matrix (word-graph)')
    matrix, vec = word_graph.make_matrix(co_occur, word_occur, context_occur, args.model)

    # save data (matrix) 
    util.trace('save the matrix')
    util.save_data(matrix, args.outF+'/pmi_matrix_{}.pickle'.format(args.model))
    util.save_data(vec, args.outF+'/pmi_vectorizer_{}.pickle'.format(args.model))

    # get the intial vector
    HITS_obj = hits.HITS(matrix)

    # matrix is symmetry; authority score is equal to hubness score.
    util.trace('start HITS')
    i = HITS_obj.startHITS(args.iter).toarray() 
    util.trace('finish HITS')

    # write the ranking words by HITS
    util.trace('write the vocabulary')
    util.writeVocab(HITS_obj, i, vocabSize, args.outF+'/vocab_file.hits')
    
    util.trace('finish program')

if __name__ == '__main__':
    main()
