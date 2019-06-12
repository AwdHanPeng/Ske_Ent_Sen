from torch.utils.data import Dataset as DS
import pickle
import torch

'''
    分别构建了训练和验证用数据集dataset类
'''

class TrainDataset(DS):
    '''
       utterance:(self.batch_size, self.max_num_utterance, self.max_sentence_len)
       response:(self.batch_size, self.max_sentence_len)
       label:(self.batch_size)
       root 中包含三个文件路径
            utterance_path:应返回list(50W,max_num_utterance,max_sentence_len)
            correct_response_path：应返回list(50W,max_sentence_len)
            error_response_path:应返回list(50W,max_sentence_len)

        注意：经过dataloader操作之后 数据全部会自动变成tensor 根本不需要自己显式的操作
    '''

    def __init__(self, root, transforms=None):
        with open('../Data/entities.pkl', 'rb') as f:
            self.entities = pickle.load(f, encoding='bytes')
        utterance_path, correct_response_path, error_response_path = root
        with open(utterance_path, 'rb') as f:
            self.utterance = pickle.load(f, encoding='bytes')
        with open(correct_response_path, 'rb') as f:
            self.correct_response = pickle.load(f, encoding='bytes')
        with open(error_response_path, 'rb') as f:
            self.error_response = pickle.load(f, encoding='bytes')
        if transforms is not None:
            self.padding_sentence, self.padding_utterance = transforms

    def generate_ske(self, lis):
        temp_lis = [item if item not in self.entities else 0 for item in lis]
        return temp_lis

    def generate_entities(self, lis):
        temp_lis = [item if item in self.entities else 0 for item in lis]
        return temp_lis

    def __len__(self):
        return len(self.utterance) * 2

    '''
        label = 1 表示属于第一类 第一类是正类
        label = 0 表示属于第〇类 第〇类是负类
        在预测时，输出则包含分别属于第一类和第〇类的概率
    '''

    def __getitem__(self, idx):
        utterance = self.utterance[int(idx / 2)]
        # idx 为奇数传正样本 为偶数传负样本
        if idx % 2 == 0:
            response = self.correct_response[int(idx / 2)]
            label = 1
        else:
            response = self.error_response[int(idx / 2)]
            label = 0
        utter_ske = [self.generate_ske(item) for item in utterance]
        utter_ent = [self.generate_entities(item) for item in utterance]
        resp_ske = self.generate_ske(response)
        resp_ent = self.generate_entities(response)
        if self.padding_sentence is not None:
            self.padding_utterance(utterance)
            self.padding_utterance(utter_ske)
            self.padding_utterance(utter_ent)
            for sentence, skeleton, entity in zip(utterance, utter_ske, utter_ent):
                self.padding_sentence(sentence)
                self.padding_sentence(skeleton)
                self.padding_sentence(entity)
            self.padding_sentence(response)
            self.padding_sentence(resp_ske)
            self.padding_sentence(resp_ent)
        utter_ske_mask = [[1 if x is 0 else 0 for x in item] for item in utter_ske]
        utter_ent_mask = [[1 if x is 0 else 0 for x in item] for item in utter_ent]
        resp_ent_mask = [1 if item is 0 else 0 for item in resp_ent]
        resp_ske_mask = [1 if item is 0 else 0 for item in resp_ske]

        return torch.tensor(utterance), torch.tensor(response), torch.tensor(label), torch.tensor(
            utter_ske), torch.tensor(resp_ske), torch.tensor(utter_ent), torch.tensor(resp_ent), torch.tensor(
            utter_ske_mask), torch.tensor(utter_ent_mask), torch.tensor(resp_ske_mask), torch.tensor(resp_ent_mask)

class ValidDataset(DS):
    '''
       utterance:(self.batch_size, self.max_num_utterance, self.max_sentence_len)
       responses:(self.batch_size, 10, self.max_sentence_len)
       correct_index:(self.batch_size)

       root 中包含三个文件路径
            utterance_path:应返回list(1W,max_num_utterance,max_sentence_len)
            responses_path：应返回list(1W,10,max_sentence_len)
            index_path:应返回list(1W)
    '''

    def __init__(self, root, transforms=None):
        with open('../Data/entities.pkl', 'rb') as f:
            self.entities = pickle.load(f, encoding='bytes')
        utterance_path, response_path, index_path = root
        with open(utterance_path, 'rb') as f:
            self.utterance = pickle.load(f, encoding='bytes')
        with open(response_path, 'rb') as f:
            self.responses = pickle.load(f, encoding='bytes')
        with open(index_path, 'rb') as f:
            self.index = pickle.load(f, encoding='bytes')
        if transforms is not None:
            self.padding_sentence, self.padding_utterance = transforms

    def generate_ske(self, lis):
        temp_lis = [item if item not in self.entities else 0 for item in lis]
        return temp_lis

    def generate_entities(self, lis):
        temp_lis = [item if item in self.entities else 0 for item in lis]
        return temp_lis

    def __len__(self):
        return len(self.utterance)

    def __getitem__(self, idx):
        utterance = self.utterance[idx]
        responses = self.responses[idx]
        correct_index = self.index[idx]
        utter_ske = [self.generate_ske(item) for item in utterance]
        resps_ske = [self.generate_ske(item) for item in responses]
        utter_ent = [self.generate_entities(item) for item in utterance]
        resps_ent = [self.generate_entities(item) for item in responses]
        if self.padding_sentence is not None:
            self.padding_utterance(utterance)
            self.padding_utterance(utter_ske)
            self.padding_utterance(utter_ent)
            for sentence, skeleton, entity in zip(utterance, utter_ske, utter_ent):
                self.padding_sentence(sentence)
                self.padding_sentence(skeleton)
                self.padding_sentence(entity)
            for sentence, skeleton, entity in zip(responses, resps_ske, resps_ent):
                self.padding_sentence(sentence)
                self.padding_sentence(skeleton)
                self.padding_sentence(entity)
        utter_ske_mask = [[1 if x is 0 else 0 for x in item] for item in utter_ske]
        utter_ent_mask = [[1 if x is 0 else 0 for x in item] for item in utter_ent]
        resp_ent_mask = [[1 if x is 0 else 0 for x in item] for item in resps_ent]
        resp_ske_mask = [[1 if x is 0 else 0 for x in item] for item in resps_ske]
        return torch.tensor(utterance), torch.tensor(responses), torch.tensor(correct_index), torch.tensor(
            utter_ske), torch.tensor(resps_ske), torch.tensor(utter_ent), torch.tensor(resps_ent), torch.tensor(
            utter_ske_mask), torch.tensor(utter_ent_mask), torch.tensor(resp_ske_mask), torch.tensor(resp_ent_mask),

class TestDataset(DS):
    '''
       utterance:(self.batch_size, self.max_num_utterance, self.max_sentence_len)
       responses:(self.batch_size, 10, self.max_sentence_len)

       root 中包含两个文件路径
            utterance_path:应返回list(1W,max_num_utterance,max_sentence_len)
            responses_path：应返回list(1W,10,max_sentence_len)
    '''

    def __init__(self, root, transforms=None):
        with open('../Data/entities.pkl', 'rb') as f:
            self.entities = pickle.load(f, encoding='bytes')
        utterance_path, response_path = root
        with open(utterance_path, 'rb') as f:
            self.utterance = pickle.load(f, encoding='bytes')
        with open(response_path, 'rb') as f:
            self.responses = pickle.load(f, encoding='bytes')
        if transforms is not None:
            self.padding_sentence, self.padding_utterance = transforms

    def generate_ske(self, lis):
        temp_lis = [item if item not in self.entities else 0 for item in lis]
        return temp_lis

    def __len__(self):
        return len(self.utterance)

    def __getitem__(self, idx):
        utterance = self.utterance[idx]
        responses = self.responses[idx]
        utter_ske = [self.generate_ske(item) for item in utterance]
        resps_ske = [self.generate_ske(item) for item in responses]
        if self.padding_sentence is not None:
            self.padding_utterance(utterance)
            self.padding_utterance(utter_ske)
            for sentence, skeleton in zip(utterance, utter_ske):
                self.padding_sentence(sentence)
                self.padding_sentence(skeleton)
            for sentence, skeleton in zip(responses, resps_ske):
                self.padding_sentence(sentence)
                self.padding_sentence(skeleton)

        return torch.tensor(utterance), torch.tensor(responses), torch.tensor(utter_ske), torch.tensor(resps_ske)
