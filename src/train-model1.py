from __future__ import print_function
import sys
import math
import numpy as np
from collections import defaultdict


class IBM():
    def __init__(self, src_file, tgt_file):
        src_sents, self.src_vocab = self.read_corpus(src_file)
        tgt_sents, self.tgt_vocab = self.read_corpus(tgt_file)
        assert len(src_sents) == len(tgt_sents), '# of src = {}, # of tgt = {}, not match'.format(len(src_sents), len(tgt_sents))
        print('|E| = {}, |F| = {}'.format(len(self.src_vocab), len(self.tgt_vocab)))

        # Training
        self.theta = None
        self.train(src_sents, tgt_sents)

    def read_corpus(self, file_name):
        sents = []
        vocab = []
        for line in open(file_name):
            vals = line.strip().split()
            sents.append(vals)
            vocab.extend(vals)
        return sents, set(vocab)

    def train(self, src_sents, tgt_sents, max_iter=7):
        # (1) initialize translation probability
        print('Begin Initialization')
        print('|E| = {}, |F| = {}'.format(len(self.src_vocab), len(self.tgt_vocab)))
        self.theta = defaultdict(lambda: defaultdict(lambda: 1.0e-10))
        for t in self.tgt_vocab:
            for s in self.src_vocab:
                self.theta[t][s] = 1.0 / (len(self.tgt_vocab) + 1)

        print('Begin Training')
        for iter in range(max_iter):
            cnt_t_given_s = defaultdict(lambda: defaultdict(lambda: 0))
            cnt_s = defaultdict(lambda: 0)

            if iter % 2 == 0:
                print('iter ', iter)
            for (s_idx, s_sent) in enumerate(src_sents):
                if s_idx % 10 == 0:
                    print('  pair', s_idx)

                t_sent = tgt_sents[s_idx]
                cnt_t = defaultdict(lambda: 0.0)

                # (2) [E] C[i,j] = theta[i,j] / sum_i theta[i,j]
                for s in s_sent:
                    for t in t_sent:
                        cnt_t[t] += self.theta[t][s]

                for t in t_sent:
                    for s in s_sent:
                        normalized_cnt = self.theta[t][s] / cnt_t[t]
                        cnt_t_given_s[t][s] += normalized_cnt
                        cnt_s[s] += normalized_cnt

            print('Done 1')
            # (2) [M] theta[t][s] = count[t][s] / count[t]
            for s in self.src_vocab:
                for t in self.tgt_vocab:
                    estimate = cnt_t_given_s[t][s] / cnt_s[s]
                    self.theta[t][s] = max(estimate, 10e-10)
            print('Done 2')

            # (3) Calculate log data likelihood
            ll = 1.0
            for (s_idx, s_sent) in enumerate(src_sents):
                t_sent = tgt_sents[s_idx]

                for s in s_sent:
                    sum_t = sum([self.theta[t][s] for t in t_sent])
                    ll *= sum_t * (1.0 / (len(self.tgt_vocab) + 1))
            print("[{}] Log likelihood : {}".format(iter, -np.log(ll)))

    def align(self, alg_file):
        pass


def main():
    src_file = sys.argv[1]
    tgt_file = sys.argv[2]
    alg_file = sys.argv[3]
    ibm = IBM(src_file, tgt_file)
    #ibm.train()
    #ibm.align(alg_file)


if __name__ == '__main__':
    main()
