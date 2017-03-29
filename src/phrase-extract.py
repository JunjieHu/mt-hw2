import sys
from util import *
import math
from collections import defaultdict


def phrase_extract(bitext, align, phr_file, freq_thred=2):
    print('Begin extraction')
    cnt_st = defaultdict(float)
    idx = 0
    for (s_sent, t_sent), alg in zip(bitext, align):
        BP = phrase_extract_sents(s_sent, t_sent, alg)
        # print(idx, BP)
        for (s_phr, t_phr) in BP:
            cnt_st[(s_phr, t_phr)] += 1.0

        if idx % 5000 == 0:
            print('{} / {}'.format(idx, len(bitext)), BP)
        idx += 1

    # Filter out phrase pairs with low frequency
    print('Begin Filtering')
    cnt_st_filt = defaultdict(float)
    cnt_t = defaultdict(float)
    for (s_phr, t_phr), cnt in cnt_st.iteritems():
        if cnt < freq_thred:
            continue
        cnt_st_filt[(s_phr, t_phr)] = cnt
        cnt_t[t_phr] += cnt

    # Score[s_phr][t_phr] = cnt_st[s_phr][t_phr] / cnt_t[t_phr]
    print('Begin scoring')
    fout = open(phr_file, 'w')
    for (s_phr, t_phr), cnt in cnt_st_filt.iteritems():
        prob = cnt / cnt_t[t_phr]
        score = -math.log(prob)
        if score == 0.0: score = 0.0
        print >> fout, "%s\t%s\t%.4f" % (s_phr, t_phr, score)
    fout.close()


def phrase_extract_sents(s_sent, t_sent, align, len_thred=None):
    BP = set()
    src_align = [j for (i, j) in align]
    for i1 in range(len(t_sent)):
        for i2 in range(i1, len(t_sent)):
            TP = [j for (i, j) in align if i1 <= i <= i2]  # (i, j) tgt-src
            if len(TP) == 0:
                continue

            is_consecutive = True
            j1, j2 = min(TP), max(TP)
            for j in range(j1, j2 + 1):
                if j in TP or j not in src_align:
                    continue
                is_consecutive = False
                break
            if not is_consecutive:
                continue

            SP = [i for (i, j) in align if j1 <= j <= j2]
            SP.sort()
            if not (i1 <= SP[0] and SP[-1] <= i2):
                continue
            t_phr = " ".join(t_sent[i1: i2 + 1])
            s_phr = " ".join(s_sent[j1: j2 + 1])

            if len(t_phr) == 0 or len(s_phr) == 0:
                continue

            #if i2 + 1 - i1 <= 3 and j2 + 1 - j1 <= 3:
            BP.add((s_phr, t_phr))
            # while j1 >= 0 and all(jj != j1 for (jj, ii) in align):
            #     jp = j2
            #     while jp <=
    return BP


if __name__ == '__main__':
    src_file = sys.argv[1]
    tgt_file = sys.argv[2]
    alg_file = sys.argv[3]
    phr_file = sys.argv[4]
    bitext = read_bitext(src_file, tgt_file)
    align = read_alignment(alg_file)
    phrase_extract(bitext, align, phr_file)
