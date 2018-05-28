import pickle
import datetime
import sys
import hits

def trace(*args):
    print(datetime.datetime.now(), '...', *args, file=sys.stderr)
    sys.stderr.flush()

def save_data(obj, output_file):
    with open(output_file, 'wb') as o_f:
        pickle.dump(obj, o_f)

def load_data(input_file):
    with open(input_file, 'rb') as i_f:
        obj = pickle.load(i_f)
    return obj

def writeVocab(hits_obj, vec, size, output_file):
    o_f = open(output_file, 'w')
    o_f.write('index\tword\tscore (relative)\n')

    for index_id, (word, value) in zip(range(size),  \
            hits_obj.sort_instance(vec)):
        if index_id % 1000 == 0:
            trace('write {}word'.format(index_id))
        o_f.write('{}\t{}\t{}\n'.format(index_id, word, value))

    o_f.close()
