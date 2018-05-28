import scipy
import numpy as np

class HITS:
    def __init__(self, matrix):
        self.word_name, self.matrix = matrix

        # make initial vector (all values are 1.)
        seed_vec = np.ones((self.matrix.shape[0], 1))
        self.seed_vec = scipy.sparse.csr_matrix(seed_vec)
        # prepare the id to word dict
        self.id2word = {v:k for k,v in self.word_name.items()}

    def startHITS(self, iterator):
        # hubness scores are equal to authority scores.

        instance = self.seed_vec
        for _ in range(iterator):
            pattern = np.dot(self.matrix.T, instance)
            instance = np.dot(self.matrix, pattern)

            instance = instance/instance.sum()
            pattern = pattern/pattern.sum()

        return instance

    def sort_instance(self, vec):
        sort_vec = sorted(list(enumerate(vec)), key=lambda x:x[1], reverse=True)
        for element_vec in sort_vec:
            yield (self.id2word[element_vec[0]], element_vec[1])
