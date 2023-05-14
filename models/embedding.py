import torch
import torch.nn as nn
from transformers import BertModel, AutoModel
from transformers import logging
 
logging.set_verbosity_warning()

class Embedding(nn.Module):
    def __init__(self, max_length=90, pos_embedding_dim=5, roberta=False): # set True when using RoBerta model for cross doamin task
        nn.Module.__init__(self)

        if roberta:
            print('roberta')
            self.bert_embedding = self.bert_embedding = AutoModel.from_pretrained("allenai/biomed_roberta_base")
        else:
            print('bert')
            self.bert_embedding = BertModel.from_pretrained('bert-base-uncased')
        self.max_length = max_length

    def forward(self, inputs):
        word = inputs['word']
        x = self.bert_embedding(word, attention_mask=inputs['mask'])     
        sequence_embedding, sentence_embedding = x[0], x[1]
        return sequence_embedding, sentence_embedding
