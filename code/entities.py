import pickle
import jieba.posseg as pseg

ent_num = set()
ent_str = {'n', 'nr', 'nr1', 'nr2', 'nrj', 'nrf', 'ns', 'nsf', 'nt', 'nz', 'nl', 'ng',
           'a', 'ad', 'an', 'ag', 'al',
           'r', 'rr', 'rz', 'rzt', 'rzs', 'rzv', 'ry', 'ryt', 'rys', 'ryv', 'rg',
           'x', 'xx', 'xu',
           'm', 'mq'
           }

error_num = 0

def get_class(key):
    '''
    得到这个单词的实体类别
    '''

    word = pseg.cut(key)
    for w in word:
        if w.word != key:
            global error_num
            error_num += 1
        return w.flag

def create_entities():
    count = 0
    with open('../Data/word2index.pkl', 'rb') as f:
        word2index = pickle.load(f)
    for word in word2index:
        index = word2index[word]
        cla = get_class(word)
        if cla in ent_str:
            ent_num.add(index)
        count += 1
        print('at index {}'.format(count))
    print(error_num / len(word2index))
    with open('../Data/entities.pkl', 'wb') as f:
        pickle.dump(ent_num, f)  # 0.08352740399793195

if __name__ == '__main__':
    create_entities()

'''
str = '你吃了吗'
word = pseg.cut(str)
for w in word:
    print(w.word, w.flag)
    哪些词不应该出现在骨架中，也是应该出现在实体集合中的
    形容词 动词（这个怎么考虑？可能要，可能不要） 名词 代词
    名词：n nr nr1 nr2 nrj nrf ns nsf nt nz nl ng
    形容词： a ad an ag al
    然后可以先把这个玩意 加一层骨干的正方形

    关键cut完之后 怎么讲list
    在embedding 那个index里边 提前定好映射规则
    然后直接把某些词转换成padding
'''
'''
1. 名词 (1个一类，7个二类，5个三类)
　　名词分为以下子类：
　　　　n 名词
　　　　nr 人名
　　　　nr1 汉语姓氏
　　　　nr2 汉语名字
　　　　nrj 日语人名
　　　　nrf 音译人名
　　　　ns 地名
　　　　nsf 音译地名
　　　　nt 机构团体名
　　　　nz 其它专名
　　　　nl 名词性惯用语
　　　　ng 名词性语素
2. 时间词(1个一类，1个二类)
　　　　t 时间词
　　　　tg 时间词性语素
3. 处所词(1个一类)
　　　　s 处所词
4. 方位词(1个一类)
　　　　f 方位词
5. 动词(1个一类，9个二类)
　　　　v 动词
　　　　vd 副动词
　　　　vn 名动词
　　　　vshi 动词“是”
　　　　vyou 动词“有”
　　　　vf 趋向动词
　　　　vx 形式动词
　　　　vi 不及物动词（内动词）
　　　　vl 动词性惯用语
　　　　vg 动词性语素
6. 形容词(1个一类，4个二类)
　　　　a 形容词
　　　　ad 副形词
　　　　an 名形词
　　　　ag 形容词性语素
　　　　al 形容词性惯用语
7. 区别词(1个一类，2个二类)
　　　　b 区别词
　　　　bl 区别词性惯用语
8. 状态词(1个一类)
　　　　z 状态词
9. 代词(1个一类，4个二类，6个三类)
　　　　r 代词
　　　　rr 人称代词
　　　　rz 指示代词
　　　　rzt 时间指示代词
　　　　rzs 处所指示代词
　　　　rzv 谓词性指示代词
　　　　ry 疑问代词
　　　　ryt 时间疑问代词
　　　　rys 处所疑问代词
　　　　ryv 谓词性疑问代词
　　　　rg 代词性语素
10. 数词(1个一类，1个二类)
　　　　m 数词
　　　　mq 数量词
11. 量词(1个一类，2个二类)
　　　　q 量词
　　　　qv 动量词
　　　　qt 时量词
12. 副词(1个一类)
　　　　d 副词
13. 介词(1个一类，2个二类)
　　　　p 介词
　　　　pba 介词“把”
　　　　pbei 介词“被”
14. 连词(1个一类，1个二类)
　　　　c 连词
　　　　cc 并列连词
15. 助词(1个一类，15个二类)
　　　　u 助词
　　　　uzhe 着
　　　　ule 了 喽
　　　　uguo 过
　　　　ude1 的 底
　　　　ude2 地
　　　　ude3 得
　　　　usuo 所
　　　　udeng 等 等等 云云
　　　　uyy 一样 一般 似的 般
　　　　udh 的话
　　　　uls 来讲 来说 而言 说来
　　　　uzhi 之
　　　　ulian 连 （“连小学生都会”）
16. 叹词(1个一类)
　　　　e 叹词
17. 语气词(1个一类)
　　　　y 语气词(delete yg)
18. 拟声词(1个一类)
　　　　o 拟声词
19. 前缀(1个一类)
　　　　h 前缀
20. 后缀(1个一类)
　　　　k 后缀
21. 字符串(1个一类，2个二类)
　　　　x 字符串
　　　　xx 非语素字
　　　　xu 网址URL
22. 标点符号(1个一类，16个二类)
　　　　w 标点符号
　　　　wkz 左括号，全角：（ 〔 ［ ｛ 《 【 〖 〈 半角：( [ { <
　　　　wky 右括号，全角：） 〕 ］ ｝ 》 】 〗 〉 半角： ) ] { >
　　　　wyz 左引号，全角：“ ‘ 『
　　　　wyy 右引号，全角：” ’ 』
　　　　wj 句号，全角：。
　　　　ww 问号，全角：？ 半角：?
　　　　wt 叹号，全角：！ 半角：!
　　　　wd 逗号，全角：， 半角：,
　　　　wf 分号，全角：； 半角： ;
　　　　wn 顿号，全角：、
　　　　wm 冒号，全角：： 半角： :
　　　　ws 省略号，全角：…… …
　　　　wp 破折号，全角：—— －－ ——－ 半角：--- ----
　　　　wb 百分号千分号，全角：％ ‰ 半角：%
　　　　wh 单位符号，全角：￥ ＄ ￡ ° ℃ 半角：$
'''
'''

    {'y': 77, 'uv': 20, 'vn': 1759,名动词
     'd': 5129副词, 'nt': 1091 机构团体, 'c': 823,连词
     'k': 8, 'h': 19, 'nr': 39300人名, 'mg': 2, 'vq': 1, 'n': 68094名词, 'uj': 77, 'nrt': 2191, 'ns': 8987, 'z': 1760, 'b': 1107, 's': 560, 'ad': 145, 'eng': 2, 'rr': 2, 'mq': 78, 'tg': 80, 'ug': 76, 'uz': 23, 'ng': 456, 'vd': 4, 'vg': 213, 'i': 7686, 'g': 361, 'f': 808, 'vi': 2, 'u': 123, 'p': 1227, 'ag': 130, 'ul': 149, 'x': 1083, 'a': 6926, 'dg': 6, 'o': 323, 'v': 40592, 'q': 394, 'nz': 5079, 'ud': 64, 'df': 1, 'rg': 1, 'e': 55, 'zg': 1360, 'r': 1730, 'l': 8271, 'nrfg': 502, 'rz': 2, 't': 1942, 'yg': 20, 'm': 6563, 'j': 1036, 'an': 43}
'''
'''
    'vn': 1759名动词
    'd': 5129副词
    'nt': 1091 机构团体
    'c': 823,连词
    'nr': 39300人名
    'n': 68094名词
    'nrt': 2191
    'ns': 8987地名
    'z': 1760状态词
    'b': 1107区别词
    's': 560处所词
    'ad': 145副形词
    'ng': 456名词性语素
    'vg': 213
    'i': 7686
    'g': 361
    'f': 808
    'u': 123, 
    'p': 1227, 
    'ag': 130, 
    'ul': 149, 
    'x': 1083, 
    'a': 6926
    'v': 40592, 动词
    'q': 394, 
    'nz': 5079
    'zg': 1360, 
    'r': 1730, 
    'l': 8271, 
    'nrfg': 502
    't': 1942
    'm': 6563, 
    'j': 1036
'''
