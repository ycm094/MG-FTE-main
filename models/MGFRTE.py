import torch
from torch import nn
import models.embedding as embedding
from torchcrf import CRF
from models.utils import rel_accuracy, extract_entity, bert_token_merge
import torch.nn.functional as F
from transformers import AutoTokenizer
import numpy as np


class MGFRTE(nn.Module):

    def __init__(self, max_length, word_embedding_dim=768, args=None, hidden_size=100, drop=True, roberta=False):
        nn.Module.__init__(self)
        self.word_embedding_dim = word_embedding_dim
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.args = args

        self.drop = drop
        self.dropout = nn.Dropout(0.2)

        self.proj = nn.Linear(self.word_embedding_dim * 4, self.word_embedding_dim)

        self.multilayer_relation = nn.Sequential(nn.Linear(self.word_embedding_dim*4, self.word_embedding_dim),
                                            nn.ReLU(),
                                            nn.Linear(self.word_embedding_dim, 1))

        self.num_entity_labels = 5
        self.CELoss = nn.CrossEntropyLoss()
        self.ner = CRF(num_tags=5, batch_first=True)
        self.embedding = embedding.Embedding()
        self.roberta = roberta
        if self.roberta:
            self.tokenizer = AutoTokenizer.from_pretrained("allenai/biomed_roberta_base")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def context_encoder(self, input, entity_label=None):
        input_mask = (input['mask'] != 0).float()
        max_length = input_mask.long().sum(1).max().item()
        input_mask = input['mask'] != 0
        input_mask = input_mask[:, :max_length].contiguous()
        embedding, sent_embedding = self.embedding(input)
        embedding_ = embedding[:, :max_length, :].contiguous()
        if entity_label != None:
            entity_label = entity_label[:, :max_length, :].contiguous()  # (BNK, L, num_entity_labels)
        else:
            entity_label = None

        if self.drop:                                          
            embedding_ = self.dropout(embedding_)

        embedding_ = embedding_.transpose(1, 2).contiguous() * input_mask.unsqueeze(1)
        embedding_ = embedding_.transpose(1, 2).contiguous()
        return embedding_, sent_embedding, input_mask, entity_label, max_length

    def cal_f1(self, entity_predict, entity_label, sent_set, entity_set, rel_predict, rel_label):
        entity_index_predict = extract_entity(entity_predict)
        entity_index_label = extract_entity(entity_label)
        gold_num = len(entity_label)
        correct_num = 0
        predict_num = 1e-10
        correct_num_triple = 0
        result = []

        for i in range(gold_num):
            sent_head_predict_list, sent_tail_predict_list = entity_index_predict[i]
            sent_head_label_list, sent_tail_label_list = entity_index_label[i]
            sent_head_predict_list, sent_tail_predict_list = list(set(sent_head_predict_list)), list(set(sent_tail_predict_list))
            sent = list(sent_set[i])
            head_gt, tail_gt = entity_set[i][0], entity_set[i][1]
            if len(sent_head_label_list) !=0 and len(sent_tail_label_list) != 0:
                head_label, tail_label = bert_token_merge(sent_head_label_list[0], sent), bert_token_merge(sent_tail_label_list[0], sent)
            else:
                head_label, tail_label = head_gt, tail_gt
            for head_index_predict in sent_head_predict_list:
                head_predict = bert_token_merge(head_index_predict, sent)
                dist, tail_predict = 9999, ''
                for tail_index_predict in sent_tail_predict_list:
                    if abs(tail_index_predict[0] - head_index_predict[0]) < dist:
                        dist = abs(tail_index_predict[0] - head_index_predict[0])
                        tail_predict = bert_token_merge(tail_index_predict, sent)

                if len(head_predict) !=0 and len(tail_predict) != 0:
                    predict_num += 1
                    result.append([' '.join(sent), head_predict, head_gt, tail_predict, tail_gt])
                    if head_predict == head_label and tail_predict == tail_label:
                        correct_num += 1
                        if rel_label[i] == rel_predict[i]:
                            correct_num_triple += 1
        precision = correct_num / predict_num
        recall = correct_num / gold_num
        f1_score = 2 * precision * recall / (precision + recall + 1e-10)

        relation_f1 = rel_accuracy(rel_predict, rel_label)

        triple_precision = correct_num_triple / predict_num
        triple_recall = correct_num_triple / gold_num
        triple_f1 = 2 * triple_precision * triple_recall / (triple_precision + triple_recall + 1e-10)
        return precision, recall, f1_score, relation_f1, triple_precision, triple_recall, triple_f1, result

    def CoAttention(self, support, query, support_mask, query_mask):
        att = support @ query.transpose(1, 2)
        att = att + support_mask * query_mask.transpose(1, 2) * 100
        support_ = F.softmax(att, 2) @ query * support_mask
        query_ = F.softmax(att.transpose(1,2), 2) @ support * query_mask
        return support_, query_

    def local_matching(self, support, query, support_mask, query_mask):  
        support_, query_ = self.CoAttention(support, query, support_mask, query_mask)
        enhance_query = self.fuse(query, query_, 2)
        enhance_support = self.fuse(support, support_, 2)
        return enhance_support, enhance_query
    
    def fuse(self, m1, m2, dim):
        return torch.cat([m1, m2, torch.abs(m1 - m2), m1 * m2], dim)

    def local_aggregation(self, enhance_support, enhance_query, support_mask, query_mask, K, support_BIO):
        # support_BIO  # (B, N, 5, D)
        max_enhance_query, _ = torch.max(enhance_query, 1)
        mean_enhance_query = torch.sum(enhance_query, 1) / torch.sum(query_mask, 1)
        enhance_query = torch.cat([max_enhance_query, mean_enhance_query], 1)

        enhance_support = enhance_support.view(enhance_support.size(0) // K, K, -1, enhance_support.shape[-1])
        support_mask = support_mask.view(enhance_support.size(0), K, -1, 1)

        max_enhance_support, _ = torch.max(enhance_support, 2)
        mean_enhance_support = torch.sum(enhance_support, 2) / torch.sum(support_mask, 2)
        enhance_support = torch.cat([max_enhance_support, mean_enhance_support], 2)

        return enhance_support, enhance_query

    def cal_bio(self, support_token_embedding, support_entity_label, support_mask, B, N, K, device):
        match_support_entity_feature = support_token_embedding  # (BNK, L, D)
        _, match_support_entity_label = torch.max(support_entity_label, -1)  # (BNK, L)
        match_tag_proto_all = torch.zeros([B*N, self.num_entity_labels, K, match_support_entity_feature.shape[-1]]).to(device)  # (BN, 5, K, 2D)
        for i in range(B*N):
            for k in range(K):
                o_feature = torch.index_select(match_support_entity_feature[i*K+k], 0, torch.nonzero((match_support_entity_label[i*K+k] == 0) * support_mask[i*K+k]).view(-1))
                bh_feature = torch.index_select(match_support_entity_feature[i*K+k], 0, torch.nonzero(match_support_entity_label[i*K+k] == 1).view(-1))
                ih_feature = torch.index_select(match_support_entity_feature[i*K+k], 0, torch.nonzero(match_support_entity_label[i*K+k] == 2).view(-1))
                bt_feature = torch.index_select(match_support_entity_feature[i*K+k], 0, torch.nonzero(match_support_entity_label[i*K+k] == 3).view(-1))
                it_feature = torch.index_select(match_support_entity_feature[i*K+k], 0, torch.nonzero(match_support_entity_label[i*K+k] == 4).view(-1))
                if len(o_feature) != 0:
                    match_tag_proto_all[i][0][k] = torch.mean(o_feature, 0)
                if len(bh_feature) != 0:
                    match_tag_proto_all[i][1][k] = torch.mean(bh_feature, 0)
                if len(ih_feature) != 0:
                    match_tag_proto_all[i][2][k] = torch.mean(ih_feature, 0)
                if len(bt_feature) != 0:
                    match_tag_proto_all[i][3][k] = torch.mean(bt_feature, 0)
                if len(it_feature) != 0:
                    match_tag_proto_all[i][4][k] = torch.mean(it_feature, 0)
        match_tag_proto_all = torch.mean(match_tag_proto_all.view(B, N, self.num_entity_labels, K, -1), -2)  # (B, N, 5, D)
        return match_tag_proto_all

    def forward_relation(self, support_token_embedding, query_token_embedding, support_mask, query_mask, B, N, K, Q, support_len, query_len, support_entity_label, device):
        
        support_BIO = self.cal_bio(support_token_embedding, support_entity_label, support_mask, B, N, K, device)  # (B, N, 5, D)
        support = support_BIO.view(B, 1, N, self.num_entity_labels, -1).repeat(1, N*Q, 1, 1, 1).contiguous().view(B * N * Q * N, self.num_entity_labels, -1)
        support_mask = torch.ones((B * N * Q * N, self.num_entity_labels, 1)).to(device)  # (BNQN, KLs, 1)
        query = query_token_embedding.view(B, N * Q, 1, query_len, -1).repeat(1, 1, N, 1, 1).contiguous().view(B * N * Q * N, query_len, -1)  # (BNQN, Lq, D)
        query_mask = query_mask.view(B, N * Q, 1, query_len).expand(B, N * Q, N, query_len).contiguous().view(-1, query_len, 1)  # (BNQN, Lq, 1)

        enhance_support, enhance_query = self.local_matching(support, query, support_mask, query_mask)  # (BNQN, 5, 4D)  (BNQN, Lq, 4D)

        # reduce dimensionality
        enhance_support = self.proj(enhance_support)  # (BNQN, KLs, D) --> (BNQN, 5, D)
        enhance_query = self.proj(enhance_query)  # (BNQN, Lq, D)
        enhance_support = torch.relu(enhance_support)
        enhance_query = torch.relu(enhance_query)

        match_support_token_embedding = enhance_support.view(B * N * Q, N, self.num_entity_labels, -1)
        match_query_token_embedding = enhance_query.view(B * N * Q, N, query_len, -1)

        # split operation
        enhance_support = enhance_support.view(B * N * Q * N, self.num_entity_labels, -1)  # (BNQNK, Ls, D)
        support_mask = support_mask.view(B * N * Q * N , self.num_entity_labels, 1)  # (BNQNK, Ls, 1)

        # Local aggregation
        # (BNQN, K, 2D)  (BNQN, 2D)
        enhance_support, enhance_query = self.local_aggregation(enhance_support, enhance_query, support_mask, query_mask, K, support_BIO)

        cat_seq = torch.cat([enhance_query, enhance_support.view(B*N*Q*N, enhance_support.shape[-1])], 1)  # (BNQN, 4D)
        query_rel_logits = self.multilayer_relation(cat_seq)  # (BNQN, 1)

        query_rel_logits = query_rel_logits.view(B * N * Q, N)
        _, rel_predict = torch.max(query_rel_logits, 1)

        match_beta = None
        return rel_predict, query_rel_logits, match_support_token_embedding, match_query_token_embedding, match_beta, support_BIO

    def __dist__(self, x, y):
        return torch.norm(x-y, 2, -1)

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2))

    def forward_entity(self, support_bio_proto, query_token_embedding, beta, support_mask, support_entity_label, rel_predict, rel_label, support_len, query_len, B, N, K, Q, device, isTrain):
        match_relation_label = rel_label if isTrain else rel_predict

        match_tag_proto = torch.zeros([B * N * Q, self.num_entity_labels, support_bio_proto.shape[-1]]).to(device)  # (BNQ, 5, D)
        match_query_entity_feature = torch.zeros([B * N * Q, query_len, query_token_embedding.shape[-1]]).to(device)  # (BNQ, L, D)
        for i in range(B * N * Q):
            match_tag_proto[i] = support_bio_proto[i][match_relation_label[i]]
            match_query_entity_feature[i] = query_token_embedding[i][match_relation_label[i]]

        match_entity_logits = -self.__batch_dist__(match_tag_proto, match_query_entity_feature)  # (BNQ, 5, D) (BNQ, L, D)
        return match_entity_logits

    def tokenize(self, raw_tokens, pos_head, pos_tail):
        # token -> index
        if self.roberta:
            tokens = ['<s>']
            for token in raw_tokens:
                token = token.lower()
                tokens += self.tokenizer.tokenize(token)
            tokens += ['</s>']
        else:
            tokens = ['[CLS]']
            for token in raw_tokens:
                token = token.lower()
                tokens += self.tokenizer.tokenize(token)
            tokens += ['[SEP]']
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        # padding
        while len(indexed_tokens) < self.max_length:
            if self.roberta:
                indexed_tokens.append(1)
            else:
                indexed_tokens.append(0)
        indexed_tokens = indexed_tokens[:self.max_length]

        # mask
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[:len(tokens)] = 1

        pos1_in_index, pos2_in_index = 0, 0
        return indexed_tokens, pos1_in_index - 1, pos2_in_index - 1, mask

    def adv(self, inputs):
        sent_token_embedding, sent_embedding, sent_mask, _, sent_len = self.context_encoder(inputs)  # (BN, L, D)
        
        return sent_embedding

    def forward(self, support, query, N, K, Q, support_rel_label, query_rel_label, support_entity_label,
                query_entity_label, device, isTrain=True):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        support_rel_label:
        '''
        support_token_embedding, support_sent, support_mask, support_entity_label, support_len = self.context_encoder(support,
                                                                                                      support_entity_label)  # (BNK, L_S, d=200)
        query_token_embedding, query_sent, query_mask, query_entity_label, query_len = self.context_encoder(query,
                                                                                            query_entity_label)  # (BNQ, L_Q, d=200)
 
        B = support['word'].size(0) // (N * K)

        # relation
        rel_predict, query_rel_logits, match_support_token_embedding, match_query_token_embedding, beta, support_BIO = \
            self.forward_relation(support_token_embedding, query_token_embedding, support_mask, query_mask, B, N, K, Q, support_len, query_len, support_entity_label, device)
        _, rel_label = torch.max(query_rel_label, -1)
        
        loss_relation = self.CELoss(query_rel_logits.view(B*N * Q, N), rel_label.view(B*N * Q))
        # for i in range(B):
        #     temp_loss_relation = self.CELoss(query_rel_logits.view(B, N * Q, N)[i], rel_label.view(B, N * Q)[i])
        #     loss_relation.append(temp_loss_relation)

        # residual
        temp_support = support_BIO.view(B, 1, N, self.num_entity_labels, -1).repeat(1, N * Q, 1, 1, 1).view(B * N * Q, N, self.num_entity_labels, -1)  # (BNQ, N, 5, D)
        temp_query = query_token_embedding.view(B, N * Q, 1, query_len, -1).repeat(1, 1, N, 1, 1).contiguous().view(B * N * Q, N, query_len, -1)  # (BNQ, N, Lq, D)
        match_support_token_embedding = torch.cat([match_support_token_embedding, temp_support], -1)
        match_query_token_embedding = torch.cat([match_query_token_embedding, temp_query], -1)

        # entity
        match_entity_logits = self.forward_entity(match_support_token_embedding, match_query_token_embedding, beta,
            support_mask, support_entity_label, rel_predict, rel_label, support_len, query_len, B, N, K, Q, device, isTrain)
        
        # CRF
        _, entity_label = torch.max(query_entity_label, -1)
        if isTrain:
            loss_entity = -self.ner(emissions=match_entity_logits.view(B*N*Q, query_len, -1), mask=query_mask.view(B*N*Q, query_len), tags=entity_label.view(B*N*Q, query_len), reduction='mean')            
            # for i in range(B):
            #     temp_loss_entity = -self.ner(emissions=match_entity_logits.view(B, N*Q, query_len, -1)[i], mask=query_mask.view(B, N*Q, query_len)[i], tags=entity_label.view(B, N*Q, query_len)[i], reduction='mean')
            #     loss_entity.append(temp_loss_entity)
        else:
            loss_entity = None

        entity_predict = self.ner.decode(emissions=match_entity_logits, mask=query_mask)
        entity_predict = [item + [0] * (query_len - len(item)) for item in entity_predict]
        entity_predict = torch.tensor(entity_predict).to(device)
        entity_predict = entity_predict.view(-1, query_len)  # (BNQ, L)

        if isTrain:
            relation_loss, entity_loss = loss_relation, loss_entity
            # for i in range(B):
            #     relation_loss += loss_relation[i]
            #     entity_loss += loss_entity[i]
        else:
            relation_loss, entity_loss = None, None


        return entity_predict, entity_label, entity_loss, rel_predict, rel_label, relation_loss