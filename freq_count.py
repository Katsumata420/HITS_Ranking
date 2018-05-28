import sys
import collections
"""
This script counts the words using frequency only.
Please compare the output result with Freq or PPMI.
"""

def main():
    """
    input_file: corpus or text
    output_folder: output folder, note "folder"!
    word_dict: key is a word, value is the number of the occurrence.
    """
    input_file = sys.argv[1]
    output_folder = sys.argv[2]
    word_dict = collections.defaultdict(int)
    counter = 0
    with open(input_file) as i_f:
        for line in i_f:
            for word in line.strip().split():
                word_dict[word] += 1

    with open(output_folder+'/vocab_file.freq', 'w') as o_f:
        o_f.write('index\tword\tscore (occurrence)\n')
        for k, v in sorted(sorted(word_dict.items()), key=lambda x:x[1], reverse=True):
            o_f.write('{}\t{}\t{}\n'.format(counter, k, v))
            counter += 1
    
if __name__ == '__main__':
    main()
