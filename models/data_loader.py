import json
import os
import numpy as np
import random
import torch
from transformers import AutoTokenizer


def getIns(bped, tokens, L, tokenizer):
    resL = 0
    tkL = " ".join(tokens[:L])
    bped_tkL = " ".join(tokenizer.tokenize(tkL))
    if bped.find(bped_tkL) != -1:
        resL = len(bped_tkL.split())
    else:
        tkL += " "
        bped_tkL = " ".join(tokenizer.tokenize(tkL))
        if bped.find(bped_tkL) != -1:
            resL = len(bped_tkL.split())
        else:
            raise Exception("Cannot locate the position")
    return resL + 1


def get_entity_label(max_len, index_head, index_tail, triple_tokenize):
    # B-head, I-head, B-tail, I-tail, O, X: 1, 2, 3, 4, 0
    entity_label = [0] * max_len
    if index_head[0] < max_len:
        entity_label[index_head[0]] = 1
    for i in range(index_head[0]+1, index_head[1]):
        if i < max_len:
            entity_label[i] = 2
    if index_tail[0] < max_len:
        entity_label[index_tail[0]] = 3
    for i in range(index_tail[0]+1, index_tail[1]):
        if i < max_len:
            entity_label[i] = 4
    return entity_label


class JSONFileDataLoader:
    def _load_preprocessed_file(self):
        name_prefix = '.'.join(self.file_name.split('/')[-1].split('.')[:-1])
        processed_data_dir = '_processed_data'
        if not os.path.isdir(processed_data_dir):
            return False
        word_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_word.npy')
        mask_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_mask.npy')
        length_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_length.npy')
        rel2scope_file_name = os.path.join(processed_data_dir, name_prefix + '_rel2scope.json')

        head_first_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_head_first.npy')
        tail_first_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_tail_first.npy')
        entity_label_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_entity_label.npy')

        entity_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_entity.npy')
        sent_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_sent.npy')

        if not os.path.exists(word_npy_file_name) or \
                not os.path.exists(mask_npy_file_name) or \
                not os.path.exists(length_npy_file_name) or \
                not os.path.exists(rel2scope_file_name) or \
                not os.path.exists(head_first_npy_file_name) or \
                not os.path.exists(tail_first_npy_file_name) or \
                not os.path.exists(entity_label_npy_file_name) or \
                not os.path.exists(sent_npy_file_name) or \
                not os.path.exists(entity_npy_file_name):
            return False
        print("Pre-processed files exist. Loading them...")
        self.data_word = np.load(word_npy_file_name)
        self.data_mask = np.load(mask_npy_file_name)
        self.data_length = np.load(length_npy_file_name)
        self.rel2scope = json.load(open(rel2scope_file_name))

        self.data_head_first = np.load(head_first_npy_file_name)
        self.data_tail_first = np.load(tail_first_npy_file_name)
        self.data_entity_label = np.load(entity_label_npy_file_name)

        self.data_entity = np.load(entity_npy_file_name, allow_pickle=True)
        self.data_sent = np.load(sent_npy_file_name, allow_pickle=True)

        if self.data_word.shape[1] != self.max_length:
            print("Pre-processed files don't match current settings. Reprocessing...")
            return False
        print("Finish loading")
        return True

    def __init__(self, file_name, max_length=90, case_sensitive=False, reprocess=False, is_same_domain=True, roberta=True):
        '''
        file_name: Json file storing the data in the following format
            {
                "P155": # relation id
                    [
                        {
                            "h": ["song for a future generation", "Q7561099", [[16, 17, ...]]], # head entity [word, id, location]
                            "t": ["whammy kiss", "Q7990594", [[11, 12]]], # tail entity [word, id, location]
                            "tokens": ["Hot", "Dance", "Club", ...], # sentence
                        },
                        ...
                    ],
                "P177":
                    [
                        ...
                    ]
                ...
            }
        max_length: The length that all the sentences need to be extend to.
        case_sensitive: Whether the data processing is case-sensitive, default as False.
        reprocess: Do the pre-processing whether there exist pre-processed files, default as False.
        '''
        self.file_name = file_name
        self.case_sensitive = case_sensitive
        self.max_length = max_length
        self.num_entity_labels = 5
        self.roberta = roberta

        if reprocess or not self._load_preprocessed_file():  # Try to load pre-processed files:
            # Check files
            if file_name is None or not os.path.isfile(file_name):
                raise Exception("[ERROR] Data file doesn't exist")

            # Load files
            print("Loading data file...")
            self.ori_data = json.load(open(self.file_name, "r"))
            print("Finish loading")

            # Eliminate case sensitive:
            if not case_sensitive:
                print("Elimiating case sensitive problem...")
                for relation in self.ori_data:
                    for ins in self.ori_data[relation]:
                        for i in range(len(ins['tokens'])):
                            ins['tokens'][i] = ins['tokens'][i].lower()
                print("Finish eliminating")

            # Pre-process data
            print("Pre-processing data...")
            self.instance_tot = 0
            for relation in self.ori_data:
                self.instance_tot += len(self.ori_data[relation])
            self.data_word = np.ones((self.instance_tot, self.max_length), dtype=np.int32)
            self.data_pos1 = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)
            self.data_pos2 = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)
            self.data_mask = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)
            self.data_entity_label = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)  # O, B-H, I-H, B-T, I-T: 0, 1, 2, 3, 4
            self.data_length = np.zeros((self.instance_tot), dtype=np.int32)
            self.rel2scope = {}  # left closed and right open
            self.data_entity = []
            self.data_sent = []

            self.data_head_first = np.zeros((self.instance_tot), dtype=np.int32)
            self.data_tail_first = np.zeros((self.instance_tot), dtype=np.int32)

            if self.roberta:
                tokenizer = AutoTokenizer.from_pretrained("allenai/biomed_roberta_base")
            else:
                tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


            head_num, tail_num = 0, 0

            i = 0
            for relation in self.ori_data:
                self.rel2scope[relation] = [i, i]
                for ins in self.ori_data[relation]:
                    words = ins['tokens']

                    sentence = ' '.join(words)

                    if self.roberta:
                        tokens_seq = tokenizer.tokenize('<s> ' + sentence + ' </s>')
                    else:
                        tokens_seq = tokenizer.tokenize('[CLS] ' + sentence + ' [SEP]')
                    token_ids = tokenizer.convert_tokens_to_ids(tokens_seq)

                    self.data_sent.append(tokens_seq)
                    self.data_entity.append((ins['h'][0], ins['t'][0]))

                    for j, word_id in enumerate(token_ids):
                        if j < self.max_length:
                            self.data_word[i][j] = word_id
                        else:
                            print('len:', len(token_ids))
                            break

                    triple = (tokenizer.tokenize(ins['h'][0]), tokenizer.tokenize(ins['t'][0]))
                    index_head_first = getIns(" ".join(tokens_seq), ins['tokens'], ins['h'][2][0][0], tokenizer)
                    index_head_last = getIns(" ".join(tokens_seq), ins['tokens'], ins['h'][2][0][-1] + 1, tokenizer)
                    index_tail_first = getIns(" ".join(tokens_seq), ins['tokens'], ins['t'][2][0][0], tokenizer)
                    index_tail_last = getIns(" ".join(tokens_seq), ins['tokens'], ins['t'][2][0][-1] + 1, tokenizer)
                    self.data_entity_label[i] = get_entity_label(self.max_length, (index_head_first, index_head_last), (index_tail_first, index_tail_last), triple)
                
                    self.data_head_first[i] = min(index_head_first, max_length)
                    self.data_tail_first[i] = min(index_tail_first, max_length)

                    self.data_mask[i][0:min(self.max_length, len(token_ids))] = 1
                    self.data_length[i] = min(len(tokens_seq), self.max_length)

                    if np.sum(self.data_entity_label[i] == 1) > 0:
                        head_num += 1
                    if np.sum(self.data_entity_label[i] == 3) > 0:
                        tail_num += 1

                    i += 1
                self.rel2scope[relation][1] = i

            print('********:', head_num, tail_num)

            print("Finish pre-processing")
            print("Storing processed files...")
            name_prefix = '.'.join(file_name.split('/')[-1].split('.')[:-1])
            processed_data_dir = '_processed_data'
            if not os.path.isdir(processed_data_dir):
                os.mkdir(processed_data_dir)

            np.save(os.path.join(processed_data_dir, name_prefix + '_word.npy'), self.data_word)
            np.save(os.path.join(processed_data_dir, name_prefix + '_mask.npy'), self.data_mask)
            np.save(os.path.join(processed_data_dir, name_prefix + '_length.npy'), self.data_length)
            json.dump(self.rel2scope, open(os.path.join(processed_data_dir, name_prefix + '_rel2scope.json'), 'w'))

            np.save(os.path.join(processed_data_dir, name_prefix + '_head_first.npy'), self.data_head_first)
            np.save(os.path.join(processed_data_dir, name_prefix + '_tail_first.npy'), self.data_tail_first)
            np.save(os.path.join(processed_data_dir, name_prefix + '_entity_label.npy'), self.data_entity_label)

            self.data_sent = np.array(self.data_sent, dtype='object')
            self.data_entity = np.array(self.data_entity, dtype=np.dtype([('head', 'U40'), ('tail', 'U40')]))
            np.save(os.path.join(processed_data_dir, name_prefix + '_entity.npy'), self.data_entity)
            np.save(os.path.join(processed_data_dir, name_prefix + '_sent.npy'), self.data_sent)

            print("Finish storing")

    def next_one(self, N=5, K=5, Q=100):
        target_classes = random.sample(self.rel2scope.keys(), N)
        support_set = {'word': [], 'mask': [], 'head_first': [], 'tail_first': []}
        query_set = {'word': [], 'mask': [], 'head_first': [], 'tail_first': []}
        support_rel_label = np.zeros([N, K, N])
        query_rel_label = np.zeros([N, Q, N])
        support_entity_label = np.zeros([N, K, self.max_length, self.num_entity_labels])
        query_entity_label = np.zeros([N, Q, self.max_length, self.num_entity_labels])

        support_sent_set, query_sent_set = [], []
        support_entity_set, query_entity_set = [], []

        for i, class_name in enumerate(target_classes):
            scope = self.rel2scope[class_name]
            indices = np.random.choice(list(range(scope[0], scope[1])), K + Q, False)
            word = self.data_word[indices]
            mask = self.data_mask[indices]
            entity_label = self.data_entity_label[indices]
            head_first = self.data_head_first[indices]
            tail_first = self.data_tail_first[indices]
            sent = self.data_sent[indices]
            entity = self.data_entity[indices]

            support_word, query_word = np.split(word, [K])
            support_mask, query_mask = np.split(mask, [K])
            support_entity_label_temp, query_entity_label_temp = np.split(entity_label, [K])
            support_head_first, query_head_first = np.split(head_first, [K])  # (K), (Q)
            support_tail_first, query_tail_first = np.split(tail_first, [K])
            support_sent, query_sent = np.split(sent, [K])
            support_entity, query_entity = np.split(entity, [K])

            support_set['word'].append(support_word)
            support_set['mask'].append(support_mask)
            query_set['word'].append(query_word)
            query_set['mask'].append(query_mask)
            support_sent_set.extend(support_sent)
            query_sent_set.extend(query_sent)
            support_entity_set.extend(support_entity)
            query_entity_set.extend(query_entity)

            support_set['head_first'].append(support_head_first)
            support_set['tail_first'].append(support_tail_first)
            query_set['head_first'].append(query_head_first)
            query_set['tail_first'].append(query_tail_first)


            for k in range(K):
                support_rel_label[i][k][i] = 1
                for l in range(self.max_length):
                    support_entity_label[i][k][l][support_entity_label_temp[k][l]] = 1

            for q in range(Q):
                query_rel_label[i][q][i] = 1
                for l in range(self.max_length):
                    query_entity_label[i][q][l][query_entity_label_temp[q][l]] = 1

        support_set['word'] = np.stack(support_set['word'], 0)
        support_set['mask'] = np.stack(support_set['mask'], 0)
        query_set['word'] = np.concatenate(query_set['word'], 0)
        query_set['mask'] = np.concatenate(query_set['mask'], 0)
        support_set['head_first'] = np.stack(support_set['head_first'], 0)  # (N, K)
        support_set['tail_first'] = np.stack(support_set['tail_first'], 0)
        query_set['head_first'] = np.stack(query_set['head_first'], 0)
        query_set['tail_first'] = np.stack(query_set['tail_first'], 0)

        query_sent_set = np.array(query_sent_set, dtype='object')
        support_entity_set = np.array(support_entity_set)  # (NK)
        query_entity_set = np.array(query_entity_set)

        return support_set, query_set, support_rel_label, query_rel_label, support_entity_label, query_entity_label, \
               query_sent_set, query_entity_set

    def next_batch(self, device, B=4, N=20, K=5, Q=100):
        support = {'word': [], 'mask': [], 'head_first': [], 'tail_first': []}
        query = {'word': [], 'mask': [], 'head_first': [], 'tail_first': []}
        support_rel_label = []
        query_rel_label = []
        support_entity_label = []
        query_entity_label = []

        query_sent, query_entity = [], []

        for one_sample in range(B):
            current_support, current_query, current_support_rel_label, current_query_rel_label, \
            current_support_entity_label, current_query_entity_label, current_query_sent, \
            current_query_entity = self.next_one(N, K, Q)

            support['word'].append(current_support['word'])
            support['mask'].append(current_support['mask'])
            query['word'].append(current_query['word'])
            query['mask'].append(current_query['mask'])
            support_rel_label.append(current_support_rel_label)  # (B, N, K, N)
            query_rel_label.append(current_query_rel_label)  # (B, N, Q, N)
            support_entity_label.append(current_support_entity_label)
            query_entity_label.append(current_query_entity_label)

            query_sent.extend(current_query_sent)
            query_entity.extend(current_query_entity)

            support['head_first'].append(current_support['head_first'])
            support['tail_first'].append(current_support['tail_first'])
            query['head_first'].append(current_query['head_first'])
            query['tail_first'].append(current_query['tail_first'])

        # (BNK, L)
        support['word'] = torch.from_numpy(np.stack(support['word'], 0)).long().view(-1, self.max_length)
        support['mask'] = torch.from_numpy(np.stack(support['mask'], 0)).long().view(-1, self.max_length)
        # (BNQ, L)
        query['word'] = torch.from_numpy(np.stack(query['word'], 0)).long().view(-1, self.max_length)
        query['mask'] = torch.from_numpy(np.stack(query['mask'], 0)).long().view(-1, self.max_length)

        support_rel_label = torch.from_numpy(np.stack(support_rel_label, 0)).float().view(-1, N)  # (BNK, N)
        query_rel_label = torch.from_numpy(np.stack(query_rel_label, 0)).float().view(-1, N)  # (BNQ, N)

        support_entity_label = torch.from_numpy(np.stack(support_entity_label, 0)).float().view(-1, self.max_length, self.num_entity_labels)  # (BNK, L, num_label)
        query_entity_label = torch.from_numpy(np.stack(query_entity_label, 0)).float().view(-1, self.max_length, self.num_entity_labels) # (BNQ, L, num_label)

        query_sent = np.array(query_sent, dtype='object')
        query_entity = np.array(query_entity)

        support['head_first'] = torch.from_numpy(np.stack(support['head_first'], 0)).long()  # (B, N, K)
        support['tail_first'] = torch.from_numpy(np.stack(support['tail_first'], 0)).long()
        query['head_first'] = torch.from_numpy(np.stack(query['head_first'], 0)).long()
        query['tail_first'] = torch.from_numpy(np.stack(query['tail_first'], 0)).long()

        for key in support:
            support[key] = support[key].to(device)
        for key in query:
            query[key] = query[key].to(device)
        support_rel_label = support_rel_label.to(device)
        query_rel_label = query_rel_label.to(device)
        support_entity_label = support_entity_label.to(device)
        query_entity_label = query_entity_label.to(device)

        return support, query, support_rel_label, query_rel_label, support_entity_label, query_entity_label, \
               query_sent, query_entity


if __name__ == '__main__':
    train_data_loader = JSONFileDataLoader('./data/fewrel/train.json', './data/glove.6B.50d.json', max_length=40, reprocess=False)
    val_data_loader = JSONFileDataLoader('./data/fewrel/val.json', './data/glove.6B.50d.json', max_length=40, reprocess=False)
    train_data_loader.next_batch(4, 20, 5, 5)
