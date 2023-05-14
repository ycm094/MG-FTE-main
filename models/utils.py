import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn


def bert_token_merge(entity_index, sent, roberta=False):
    '''
    :param entity: str, split by " "
    :return: str
    '''
    if len(entity_index) == 0:
        return ""
    entity_token_list = sent[entity_index[0]: entity_index[1]]
    
    """
    if roberta:
        entity = ' '.join(entity_token_list)
    else:
        entity_token_list = sent[entity_index[0]: entity_index[1]]
        entity = []
        for token in entity_token_list:
            if token.startswith('##'):
                token = token.lstrip('##')
                if len(entity) > 0:
                    entity[-1] = entity[-1] + token  # 同一个词被bert分成多个token
                else:
                    entity.append(token)
            else:
                entity.append(token)
        entity = " ".join(entity)
    # print(entity)
    """
    entity = ' '.join(entity_token_list)
    return entity


def rel_accuracy(rel_predict, rel_label):
    return torch.mean((rel_predict == rel_label).float()).item()


def extract_entity(query_entity_predict):
    '''
    :param query_entity_predict: (BNQ, L)
    '''
    entity_pair = []
    # [([(s1, e1), (s2, e2), ...], [(s1, e1), (s2, e2), ...]), ([], []), ...]
    for j, sent in enumerate(query_entity_predict):
        i = 0
        head_index, tail_index = [], []  # [(s1, e1), (s2, e2), ...] 左闭右开
        while i < len(sent):
            if sent[i] == 1:
                start_index = i
                i += 1
                while i < len(sent) and sent[i] == 2:
                    i += 1
                end_index = i
                head_index.append((start_index, end_index))
            elif sent[i] == 3:
                start_index = i
                i += 1
                while i < len(sent) and sent[i] == 4:
                    i += 1
                end_index = i
                tail_index.append((start_index, end_index))
            else:
                i += 1
        entity_pair.append((head_index, tail_index))
    return entity_pair
