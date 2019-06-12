from torch import nn
import torch.nn.functional as F
import torch
import pickle

'''
    此文件构架模型并实现正向传播过程
'''

'''
    Embedding:
        torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, weight=None)
        input:LongTensor of arbitrary shape containing the indices to extract
        Output: (*, embedding_dim), where * is the input shape
    GRU:
        torch.nn.GRU(*args, **kwargs)
        input_size hidden_size 
        batch_first – If True, then the input and output tensors are provided as (batch, seq, feature)
        inputs: 
                input (seq_len, batch, input_size)
                h_0  (num_layers * num_directions, batch, hidden_size)
        outputs: 
                output (seq_len, batch, num_directions * hidden_size)
                h_n (num_layers * num_directions, batch, hidden_size)
    Conv2d:
        torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        input: N, C, H, W
        output: N, C, H, W (N is batch_size, C is channel)
'''

class Config():
    def __init__(self,type):
        self.max_num_utterance = 10
        self.max_sentence_len = 20
        self.word_embedding_size = 200
        self.GRU1_hidden_size = 200  # GRU1的hidden size
        self.GRU2_hidden_size = 50  # GRU2的hidden size
        self.total_words = 218563
        self.v_length = 50
        self.type = type

embedding_file = '../Data/embedding/final_embedding.pkl'

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.max_num_utterance = config.max_num_utterance
        self.max_sentence_len = config.max_sentence_len
        self.word_embedding_size = config.word_embedding_size
        self.GRU1_hidden_size = config.GRU1_hidden_size
        self.GRU2_hidden_size = config.GRU2_hidden_size
        self.total_words = config.total_words
        self.v_length = config.v_length
        self.type = config.type
        self.word_embedding = nn.Embedding(num_embeddings=self.total_words, embedding_dim=self.word_embedding_size)
        with open(embedding_file, 'rb') as f:
            embedding_matrix = pickle.load(f, encoding="bytes")  # list
            embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)
            assert embedding_matrix.shape == (218563, 200)
        self.word_embedding.weight = nn.Parameter(embedding_matrix)
        self.word_embedding.weight.requires_grad = True

        self.sentence_GRU = nn.GRU(input_size=self.word_embedding_size, hidden_size=self.GRU1_hidden_size,
                                   bidirectional=False, batch_first=True)

        self.skeleton_GRU = nn.GRU(input_size=self.word_embedding_size, hidden_size=self.GRU1_hidden_size,
                                   bidirectional=False, batch_first=True)

        self.entity_GRU = nn.GRU(input_size=self.word_embedding_size, hidden_size=self.GRU1_hidden_size,
                                 bidirectional=False, batch_first=True)

        self.conv1 = nn.Conv2d(2, 8, kernel_size=(3, 3))

        self.conv2 = nn.Conv2d(2, 8, kernel_size=(3, 3))

        self.conv3 = nn.Conv2d(2, 8, kernel_size=(3, 3))

        self.conv4 = nn.Conv2d(2, 8, kernel_size=(3, 3))

        self.conv5 = nn.Conv2d(2, 8, kernel_size=(3, 3))

        self.pool = nn.MaxPool2d((3, 3), stride=(3, 3))
        self.linear = nn.Linear(288, self.v_length)

        self.final_GRU = nn.GRU(input_size=self.v_length, hidden_size=self.GRU2_hidden_size, bidirectional=False,
                                batch_first=True)

        self.W_1_1 = torch.nn.Parameter(
            torch.randn(self.GRU2_hidden_size, self.GRU1_hidden_size))

        self.W_1_2 = torch.nn.Parameter(
            torch.randn(self.GRU2_hidden_size, self.GRU2_hidden_size))

        self.b_1 = torch.nn.Parameter(torch.randn(self.GRU2_hidden_size))

        self.t_s = torch.nn.Parameter(torch.randn(self.GRU2_hidden_size))

        self.final_linear = nn.Linear(self.GRU2_hidden_size * 1, 2)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.GRU):
                w = (param.data for name, param in m.named_parameters() if 'weight' in name)
                for k in w:
                    nn.init.orthogonal_(k)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias)

    def forward(self, utterance, response, utter_ske, resp_ske, utter_ent, resp_ent, utter_ske_mask, utter_ent_mask,
                resp_ske_mask, resp_ent_mask):
        '''
           utterance:(self.batch_size, self.max_num_utterance, self.max_sentence_len)
           response:(self.batch_size, self.max_sentence_len)
           utter_ske:(self.batch_size, self.max_num_utterance, self.max_sentence_len)
           res_ske:(self.batch_size, self.max_sentence_len)
        '''

        utter_ske_mask = utter_ske_mask.permute(1, 0, 2)
        utter_ent_mask = utter_ent_mask.permute(1, 0, 2)

        all_utter_ske_embeddings = self.word_embedding(utter_ske)
        all_utter_ent_embeddings = self.word_embedding(utter_ent)
        reps_ske_embeddings = self.word_embedding(resp_ske)
        reps_ent_embeddings = self.word_embedding(resp_ent)
        all_utterance_embeddings = self.word_embedding(utterance)
        # batch_size,max_num_utterance,max_sentence_len ->
        # batch_size,max_num_utterance,max_sentence_len,word_embedding_size

        response_embeddings = self.word_embedding(response)
        # batch_size,max_sentence_len -> batch_size,max_sentence_len,word_embedding_size

        all_utterance_embeddings = all_utterance_embeddings.permute(1, 0, 2, 3)
        # batch_size,max_num_utterance,max_sentence_len ->
        # max_num_utterance,batch_size,max_sentence_len,word_embedding_size

        all_utter_ske_embeddings = all_utter_ske_embeddings.permute(1, 0, 2, 3)
        all_utter_ent_embeddings = all_utter_ent_embeddings.permute(1, 0, 2, 3)

        response_GRU_embeddings, _ = self.sentence_GRU(response_embeddings)
        # batch_size,max_sentence_len,word_embedding_size ->
        # batch_size,max_sentence_len,GRU1_hidden_size

        resp_ske_GRU_embeddings, _ = self.skeleton_GRU(reps_ske_embeddings)
        resp_ent_GRU_embeddings, _ = self.entity_GRU(reps_ent_embeddings)
        x1_matching_vectors = []
        x2_matching_vectors = []
        x3_matching_vectors = []
        x4_matching_vectors = []
        x5_matching_vectors = []
        for utterance_embeddings, utter_ske_embeddings, utter_ent_embeddings, utterence_ske_mask, utterence_ent_mask in zip(
                all_utterance_embeddings, all_utter_ske_embeddings, all_utter_ent_embeddings, utter_ske_mask,
                utter_ent_mask):
            # utterance_embeddings (batch_size,max_sentence_len,word_embedding_size)
            # response_embeddings (batch_size,max_sentence_len,word_embedding_size)

            ske_mask = torch.einsum('ab,ac->abc', utterence_ske_mask, resp_ske_mask)
            ent_mask = torch.einsum('ab,ac->abc', utterence_ent_mask, resp_ent_mask)

            matrix1 = torch.einsum('abe,ace->abc', utterance_embeddings, response_embeddings)
            # matrix1 (batch_size,max_sentence_len,max_sentence_len)

            utterance_GRU_embeddings, _ = self.sentence_GRU(utterance_embeddings)
            utter_ske_GRU_embeddings, _ = self.skeleton_GRU(utter_ske_embeddings)
            utter_ent_GRU_embeddings, _ = self.entity_GRU(utter_ent_embeddings)

            # utterance_GRU_embeddings (batch_size,max_sentence_len,word_embedding_size) ->
            # (batch_size,max_sentence_len,GRU1_hidden_size)
            # hidden_state (batch_size,hidden_size)
            matrix2 = torch.einsum('abe,ace->abc', utterance_GRU_embeddings, response_GRU_embeddings)
            # matrix2 (batch_size,max_sentence_len,max_sentence_len)

            matrix3 = torch.einsum('abe,ace->abc', utter_ske_GRU_embeddings, resp_ske_GRU_embeddings)
            matrix4 = torch.einsum('abe,ace->abc', utter_ske_embeddings, reps_ske_embeddings)

            matrix3.masked_fill(ske_mask.to(torch.uint8), 0)
            matrix4.masked_fill(ske_mask.to(torch.uint8), 0)

            matrix5 = torch.einsum('abe,ace->abc', utter_ent_GRU_embeddings, resp_ent_GRU_embeddings)
            matrix6 = torch.einsum('abe,ace->abc', utter_ent_embeddings, reps_ent_embeddings)

            matrix5.masked_fill(ent_mask.to(torch.uint8), 0)
            matrix6.masked_fill(ent_mask.to(torch.uint8), 0)

            matrix7 = torch.einsum('abe,ace->abc', utter_ent_GRU_embeddings, resp_ske_GRU_embeddings)
            matrix8 = torch.einsum('abe,ace->abc', utter_ent_embeddings, reps_ske_embeddings)
            matrix9 = torch.einsum('abe,ace->abc', utter_ske_GRU_embeddings, resp_ent_GRU_embeddings)
            matrix10 = torch.einsum('abe,ace->abc', utter_ske_embeddings, reps_ent_embeddings)

            matrix_sen = torch.stack([matrix1, matrix2], dim=1)
            matrix_ske = torch.stack([matrix3, matrix4], dim=1)
            matrix_ent = torch.stack([matrix5, matrix6], dim=1)
            # matrix (batch_size,2,max_sentence_len,max_sentence_len)

            matrix_cosa = torch.stack([matrix7, matrix8], dim=1)
            matrix_cosb = torch.stack([matrix9, matrix10], dim=1)

            x_conv1 = self.pool(F.relu(self.conv1(matrix_sen)))
            x_conv2 = self.pool(F.relu(self.conv2(matrix_ske)))
            x_conv3 = self.pool(F.relu(self.conv3(matrix_ent)))

            x_conv4 = self.pool(F.relu(self.conv4(matrix_cosa)))
            x_conv5 = self.pool(F.relu(self.conv5(matrix_cosb)))

            x1 = x_conv1.view(x_conv1.size(0), -1)
            x2 = x_conv2.view(x_conv2.size(0), -1)
            x3 = x_conv3.view(x_conv3.size(0), -1)
            x4 = x_conv4.view(x_conv4.size(0), -1)
            x5 = x_conv5.view(x_conv5.size(0), -1)

            x1_matching_vector = torch.tanh(self.linear(x1))
            x1_matching_vectors.append(x1_matching_vector)

            x2_matching_vector = torch.tanh(self.linear(x2))
            x2_matching_vectors.append(x2_matching_vector)

            x3_matching_vector = torch.tanh(self.linear(x3))
            x3_matching_vectors.append(x3_matching_vector)

            x4_matching_vector = torch.tanh(self.linear(x4))
            x4_matching_vectors.append(x4_matching_vector)

            x5_matching_vector = torch.tanh(self.linear(x5))
            x5_matching_vectors.append(x5_matching_vector)

        _, x1_final_hidden = self.final_GRU(torch.stack(x1_matching_vectors, dim=1))
        x1_final_hidden = torch.squeeze(x1_final_hidden)
        _, x2_final_hidden = self.final_GRU(torch.stack(x2_matching_vectors, dim=1))
        x2_final_hidden = torch.squeeze(x2_final_hidden)
        _, x3_final_hidden = self.final_GRU(torch.stack(x3_matching_vectors, dim=1))
        x3_final_hidden = torch.squeeze(x3_final_hidden)
        _, x4_final_hidden = self.final_GRU(torch.stack(x4_matching_vectors, dim=1))
        x4_final_hidden = torch.squeeze(x4_final_hidden)
        _, x5_final_hidden = self.final_GRU(torch.stack(x5_matching_vectors, dim=1))
        x5_final_hidden = torch.squeeze(x5_final_hidden)
        # (N,v_length)->(N,GRU2_hidden_size)
        if self.type == 1:
            L = x1_final_hidden
        elif self.type == 2:
            L = x2_final_hidden
        elif self.type == 3:
            L = x3_final_hidden
        elif self.type == 4:
            L = x4_final_hidden
        elif self.type == 5:
            L = x5_final_hidden
        elif self.type == 6:
            L = torch.cat((x1_final_hidden, x2_final_hidden, x3_final_hidden, x4_final_hidden, x5_final_hidden), dim=1)
        # batch_size, GRU2_hidden_size
        logits = self.final_linear(L)
        y_pred = F.log_softmax(logits, dim=1)
        y_pred_pro = F.softmax(logits, dim=1)
        return y_pred, y_pred_pro
