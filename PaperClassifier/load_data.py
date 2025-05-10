from nltk.tokenize import word_tokenize
from vocab import Vocab
from sklearn.model_selection import train_test_split


def load_data():
    f = open('data/nlp_1.csv', encoding='utf-8-sig')
    nlp_abstracts = list()
    for line in f:
        line = word_tokenize(line)
        if len(line) > 0:
            line.pop(0)
            line.pop(-1)
            nlp_abstracts.append(line)
    f.close()
    f = open('data/cv_1.csv', encoding='utf-8-sig')
    cv_abstracts = list()
    for line in f:
        line = word_tokenize(line)
        if len(line) > 0:
            line.pop(0)
            line.pop(-1)
            cv_abstracts.append(line)
    f.close()

    vocab = Vocab.build(nlp_abstracts + cv_abstracts)

    data = [(vocab.convert_tokens_to_ids(sentence), 0) for sentence in nlp_abstracts] + \
           [(vocab.convert_tokens_to_ids(sentence), 1) for sentence in cv_abstracts]
    train_data, test_data = train_test_split(data, test_size=0.2)
    return train_data, test_data, vocab
