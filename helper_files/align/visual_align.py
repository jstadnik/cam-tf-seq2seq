

def load_wmap(path, inverse=False):
    with open(path) as f:
        d = dict(line.strip().split(None, 1) for line in f)
        if inverse:
            d = dict(zip(d.values(), d.keys()))
        for (s, i) in [('<s>', '1'), ('</s>', '2')]:
            if not s in d or d[s] != i:
                logging.warning("%s has not ID %s in word map %s!" % (s, i, path))
        return d


def produce_table(sentence_in, sentence_out, alignment, wmap_in, wmap_out):
    wmap_in_dict = load_wmap(wmap_in, True)
    wmap_out_dict = load_wmap(wmap_out, True)

    sentence_in = sentence_in.strip().split()
    sentence_out = sentence_out.strip().split()

    sentence_in_word = []
    sentence_out_word = []

    for word in sentence_in:
        sentence_in_word.append(wmap_in_dict.get(word, 'unk'))

    for word in sentence_out:
        sentence_out_word.append(wmap_out_dict.get(word, 'unk'))




if len(sys.argv)<2:
    print("Not enough arguments")
else:
