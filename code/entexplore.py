import pickle

def entities():
    with open('../Data/entities.pkl', 'rb') as f:
        entities = pickle.load(f)
        return entities

def count(path, ent_set):
    sum = 0
    ent = 0
    with open(path, 'rb') as f:
        txt = pickle.load(f)
    for session in txt:
        for sentence in session:
            for word in sentence:
                if word in ent_set:
                    ent += 1
                sum += 1
    print('ent num is {},ske num is {}.sum is {}'.format(ent/sum, (sum - ent)/sum, sum))

if __name__ == '__main__':
    count('../Data/train/utterance.pkl', entities())
