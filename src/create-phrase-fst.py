__author__ = 'jjhu'
import sys
from collections import defaultdict


def create_fst(phrase_file, fst_file):
    ffst = open(fst_file, 'w')
    state = defaultdict(dict)
    state_idx = 0

    for line in open(phrase_file, 'r'):
        phr = line.strip().split('\t')
        s_phr, t_phr, score = phr[0], phr[1], phr[2]

        pre_state = 0
        for s in s_phr.strip().split():
            if (s, u"<eps>") in state[pre_state]:
                pre_state = state[pre_state][(s, u"<eps>")]
            else:
                state_idx += 1
                state[pre_state][(s, u"<eps>")] = state_idx
                print >> ffst, "{} {} {} <eps>".format(pre_state, state_idx, s)
                pre_state = state_idx

        for t in t_phr.strip().split():
            if (u"<eps>", t) in state[pre_state]:
                pre_state = state[pre_state][(u"<eps>", t)]
            else:
                state_idx += 1
                state[pre_state][(u"<eps>", t)] = state_idx
                print >> ffst, "{} {} <eps> {}".format(pre_state, state_idx, t)
                pre_state = state_idx

        print >> ffst, str(pre_state) + " 0 <eps> <eps> " + score

    print >> ffst, '0 0 </s> </s>'
    print >> ffst, '0 0 <unk> <unk>'
    print >> ffst, '0'
    ffst.close()


create_fst(sys.argv[1], sys.argv[2])