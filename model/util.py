from collections import Counter
from processors import parser

def make_vocabs (gold_data):
    """
    Generates word- and context-vocabularies given in the gold_data file.
    """
    # From Levy/Goldberg 2014: "All tokens were converted to lowercase, and words
    # and contexts that appeared less than 100 times were filtered."

    word_freq = Counter()
    context_freq = Counter()
    for sentence in parser.parse_preprocessed(gold_data):
        # Retrieve the words and contexts, and update our counters
        words, contexts = retrieve_word_contexts(sentence)
        word_freq.update(set(words))
        all_contexts = []
        for context in contexts:
            all_contexts += context # Concat

        context_freq.update(set(all_contexts))

    # Now throw out everything that appears less than 100 times and assign
    # correct indices.
    word_vocab = {
        '<pad>': 0,
        '<unk>': 1
    }
    contexts_vocab = {
        '<pad>': 0, # We need to pad the contexts to uniform length
        '<unk>': 1
    }

    for k, c in word_freq.items():
        if c >= 1: # TODO: Due to too less data, we have to reduce the threshold from the original 100.
            word_vocab[k] = len(word_vocab)

    for k, c in context_freq.items():
        if c >= 1: # TODO: Due to too less data, we have to reduce the threshold from the original 100.
            contexts_vocab[k] = len(contexts_vocab)

    return word_vocab, contexts_vocab

def retrieve_word_contexts (sentence):
    """
    Retrieves all contexts for all words in the sentence. From Levy/Goldberd 2014:
    "After parsing each sentence, we derive word contexts as follows: for a target
    word w with modifiers m_1, ..., m_k and a head h, we consider the contexts
    (m_1, lbl_1), ..., (m_k, lbl_k), (h, lbl_h-1), where lbl is the type of the
    dependency relation between the head and the modifier (e.g. nsubj, dobj,
    prep_with, amod) and lbl-1 is used to mark the inverse-relation.

    Relations that include a preposition are "collapsed" prior to context
    extraction, by directly connecting the head and the object of the preposition,
    and subsuming the preposition itself into the dependency label."

    In this function:

    w = token
    m = dependants
    lbl = token['deprel']/dependants['deprel']
    """
    words = []
    contexts = []
    for token in sentence:
        # With respect to Figure 1 in Levy/Goldberg 2014, we have to extract
        # all possible contexts for the current word/token.
        words.append(token['text'])
        internal_contexts = []
        if token['head'] > 0:
            # TODO: We have to "collapse" multiple dependencies, so we have
            # to follow ALL dependency relations to the root
            head = sentence[token['head'] - 1]
            internal_contexts.append(f"{head['text']}/{token['deprel']}-1")

        # Retrieve all dependants for this token and add them
        dependants = [dep for dep in sentence if dep['head'] == token['id']]
        for dep in dependants:
            internal_contexts.append(f"{dep['text']}/{dep['deprel']}")

        contexts.append(internal_contexts)

    return words, contexts # int, (int, num_contexts)

def find_genders (sentence, gender_map):
    """
    Returns a tuple subj, obj for the two genders in this sentence

    (0 = unknown, 1 = male, 2 = female)
    """
    subj = gender_map['unknown']
    obj = gender_map['unknown']

    # First, attempt to find the subject. There are multiple possibilities for
    # a subject (NOTE that this list is likely incomplete):

    # 1. nsubj --> root
    # 2. nsubj:pass --> root
    # 3. nsubj --> ccomp --> root
    # 4. nsubj --> advcl --> root
    # 5. det:poss --> ??? --> root
    for token in sentence:
        if token['deprel'] == 'nsubj':
            subj = gender_map[token['gender']]
            break
        elif token['deprel'] == 'nsubj:pass':
            subj = gender_map[token['gender']]
            break

    # Second, attempt to find the object. Also, multiple possibilities
    # (NOTE that this list is also likely incomplete):

    # 1. obj --> root
    # 2. iobj --> root
    # 3. obj --> xcomp --> root
    # 4. obj --> advcl --> root
    # 5. obl --> root
    # 6. ??? --> csubj:pass --> root
    # 7. nmod:poss --> case --> root
    for token in sentence:
        if token['deprel'] == 'obj':
            obj = gender_map[token['gender']]
            break
        elif token['deprel'] == 'iobj':
            obj = gender_map[token['gender']]
            break
        elif token['deprel'] == 'nmod:poss':
            obj = gender_map[token['gender']]
            break
        elif token['deprel'] == 'obl':
            obj = gender_map[token['gender']]
            break

    return subj, obj
