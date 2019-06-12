import pickle
import jieba.posseg as pseg

def get_class(key):
    '''
    得到这个单词的实体类别
    '''

    word = pseg.cut(key)
    for w in word:
        return w.flag

def count():
    dic = dict()
    with open('../Data/word2index.pkl', 'rb') as f:
        word2index = pickle.load(f)
    for word in word2index:
        cla = get_class(word)
        if cla not in dic:
            dic[cla] = 0
        dic[cla] += 1
    return dic

if __name__ == '__main__':
    dic = count()
    print(dic)
