from __future__ import print_function
import sys
import math
from collections import defaultdict
from util import *
import pickle


class IBM():
    def __init__(self, bitext, src_vocab, tgt_vocab):
        self.bitext, self.src_vocab, self.tgt_vocab = bitext, src_vocab, tgt_vocab
        self.theta = None
        self.epsilon = 1.0 / 50
        self.max_iter = 50

    def read_corpus(self, file_name):
        sents = []
        vocab = []
        for line in open(file_name):
            vals = line.strip().split()
            sents.append(vals)
            vocab.extend(vals)
        return sents, set(vocab)

    def train(self):
        # (1) initialize translation probability
        print('Begin Initialization')
        self.theta = defaultdict(lambda: defaultdict(float))
        for (s_sent, t_sent) in self.bitext:
            for t in t_sent:
                for s in s_sent:
                    self.theta[t][s] = 1.0 / (len(self.tgt_vocab) + 1)

        print('Begin Training')
        cnt_t_given_s = defaultdict(lambda: defaultdict(float))
        cnt_t = defaultdict(lambda: 0)
        for iter in range(self.max_iter):
            # cnt_t_given_s = defaultdict(lambda: defaultdict(float))
            # cnt_s = defaultdict(lambda: 0)

            cnt_t = defaultdict(lambda: 0.0)
            for (s_idx, (s_sent, t_sent)) in enumerate(self.bitext):
                if s_idx % 1000 == 0:
                    print('Iter {}, Pair {}'.format(iter, s_idx))

                # (2) [E] C[i,j] = theta[i,j] / sum_i theta[i,j]
                for s in s_sent:
                    norm = sum([self.theta[t][s] for t in t_sent])
                    for t in t_sent:
                        normalized_cnt = self.theta[t][s] / norm
                        cnt_t_given_s[t][s] += normalized_cnt
                        cnt_t[t] += normalized_cnt
            print('Done 1')

            # (2) [M] theta[t][s] = count[t][s] / count[t]
            for (t, s_dict) in self.theta.iteritems():
                for s in s_dict:
                    self.theta[t][s] = max(cnt_t_given_s[t][s] / cnt_t[t], 10e-10)
            print('Done 2')

            # (3) Calculate log data likelihood
            ll = 0.0
            for (s_sent, t_sent) in self.bitext:
                ll += math.log(self.epsilon)
                ll -= math.log(1.0 + float(len(t_sent))) * float(len(s_sent))

                for s in s_sent:
                    sum_t = sum([self.theta[t][s] for t in t_sent])
                    ll += math.log(sum_t)
            ll /= float(len(self.bitext))
            print("[{}] Log likelihood : {}".format(iter, ll))
            print(self.theta["mit"]["with"])

    def align(self, alg_file):
        fout = open(alg_file, 'w')
        for (s_sent, t_sent) in self.bitext:
            align = []
            for (tidx, t) in enumerate(t_sent):
                best_idx, best_score = -1, 0
                for (sidx, s) in enumerate(s_sent):
                    if self.theta[t][s] > best_score:
                        best_idx = sidx
                        best_score = self.theta[t][s]
                align.append("{}-{}".format(best_idx, tidx))  # src-tgt
            fout.write(" ".join(align) + '\n')

    def save(self, save_file):
        with open(save_file, 'w') as f:
            pickle.dump(self.theta, f)

    def load(self, load_file):
        with open(load_file, 'r') as f:
            self.theta = pickle.load(f)


def main():
    src_file = sys.argv[1]
    tgt_file = sys.argv[2]
    alg_file = sys.argv[3]
    bitext, src_vocab, tgt_vocab = read_bitext(src_file, tgt_file)
    ibm = IBM(bitext, src_vocab, tgt_vocab)
    ibm.train()
    ibm.align(alg_file)
    ibm.save('output/ibm1.pkl')


if __name__ == '__main__':
    main()
