import jieba.posseg as pseg
import pickle

# str = '你吃了吗'
# word = pseg.cut(str)
# for w in word:
#     print(w.word, w.flag)

'''
    哪些词不应该出现在骨架中，也是应该出现在实体集合中的
    形容词 动词（这个怎么考虑？可能要，可能不要） 名词 代词
    名词：n nr nr1 nr2 nrj nrf ns nsf nt nz nl ng
    形容词： a ad an ag al
    然后可以先把这个玩意 加一层骨干的正方形
'''
with open('../Data/entities.pkl', 'rb') as f:
    entities = pickle.load(f)
print(len(entities))
