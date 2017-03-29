from __future__ import print_function
import sys
import math
from collections import defaultdict
from util import *


class IBM1():
    def __init__(self, bitext, src_vocab, tgt_vocab):
        self.bitext = bitext
        self.src_vocab, self.tgt_vocab = src_vocab, tgt_vocab
        self.theta = None
        self.epsilon = 1.0 / max(len(ss) for (ss, tt) in self.bitext)
        self.max_iter = 20

    def train(self):
        # (1) initialize translation probability
        print('Begin Initialization')
        self.theta = defaultdict(lambda: defaultdict(float))
        cnt_ts = defaultdict(float)
        cnt_t = defaultdict(float)

        for idx, (s_sent, t_sent) in enumerate(self.bitext):
            for t in t_sent:
                cnt_t[t] += 1
                for s in s_sent:
                    cnt_ts[(t, s)] += 1

        for (s_sent, t_sent) in self.bitext:
            for t in t_sent:
                for s in s_sent:
                    self.theta[t][s] = cnt_ts[(t, s)] / cnt_t[t]

        print('Begin Training JJ')
        for iter in range(self.max_iter):
            cnt_t_given_s = defaultdict(lambda: defaultdict(float))
            cnt_t = defaultdict(float)

            for (s_idx, (s_sent, t_sent)) in enumerate(self.bitext):
                if s_idx % 20000 == 0:
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
                    # self.theta[t][s] = max(cnt_t_given_s[t][s] / cnt_t[t], 10e-20)
                    self.theta[t][s] = max(cnt_t_given_s[t][s] / cnt_t[t], 10e-20)
            print('Done 2')

            # (3) Calculate log data likelihood
            ll = 0.0
            for (s_sent, t_sent) in self.bitext:
                ll += math.log(self.epsilon)
                ll -= float(len(s_sent)) * math.log(1.0 + float(len(t_sent)))

                for s in s_sent:
                    sum_t = sum([self.theta[t][s] for t in t_sent])
                    ll += math.log(sum_t)
            ll /= float(len(self.bitext))
            print("[{}] Log likelihood : {}".format(iter, ll))
            print(self.theta["with"]["mit"])

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
                align.append("{}-{}".format(tidx, best_idx))  # tgt-src
            fout.write(" ".join(align) + '\n')


def main():
    src_file = sys.argv[1]
    tgt_file = sys.argv[2]
    alg_file = sys.argv[3]
    # src_file = "../en-de/valid.en-de.low.de"
    # tgt_file = "../en-de/valid.en-de.low.en"
    # alg_file = "../output/valid-alignment.txt"

    src_sents, src_vocab = read_corpus(src_file)
    tgt_sents, tgt_vocab = read_corpus(tgt_file)
    bitext = zip(src_sents, tgt_sents)
    ibm = IBM1(bitext, src_vocab, tgt_vocab)
    ibm.train()
    ibm.align(alg_file)


if __name__ == '__main__':
    main()
