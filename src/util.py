__author__ = 'jjhu'


def read_bitext(src_file, tgt_file):
    src_sents, src_vocab = read_corpus(src_file)
    tgt_sents, tgt_vocab = read_corpus(tgt_file)
    assert len(src_sents) == len(tgt_sents), '#src = {} != #tgt = {}'.format(len(src_sents), len(tgt_sents))
    print('|E| = {}, |F| = {}'.format(len(src_vocab), len(tgt_vocab)))
    bitext = [(s, tgt_sents[sidx]) for (sidx, s) in enumerate(src_sents)]
    return bitext, src_vocab, tgt_vocab


def read_corpus(file_name):
    sents = []
    vocab = []
    for line in open(file_name):
        vals = line.strip().split()
        sents.append(vals)
        vocab.extend(vals)
    return sents, set(vocab)


def read_alignment(file_name):
    result = []
    for line in open(file_name, 'r'):
        align = line.strip().split(' ')
        align = [tuple(map(lambda x: int(x), a.split('-'))) for a in align]
        result.append(align)
    return result
