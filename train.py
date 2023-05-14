import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

from pickle import FALSE
import random
import torch
import numpy as np
import argparse
from models.data_loader import JSONFileDataLoader
from models.framework import FewShotREFramework
from models.MGFRTE import MGFRTE as MGFRTE
from models.d import Discriminator
from models.data_loader_unsupervised import get_loader_unsupervised


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = int(3000)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
print('seed: ', seed)
torch.set_printoptions(precision=8)



parser = argparse.ArgumentParser(description='Mutially Guided Few-shot Learning for Relational Triple Extraction')
parser.add_argument('--model_name', type=str, default='MG-FRTE', help='Model name')
parser.add_argument('--N_for_train', type=int, default=5, help='Num of classes for each batch for training')
parser.add_argument('--N_for_test', type=int, default=5, help='Num of classes for each batch for testing')
parser.add_argument('--K', type=int, default=5, help='Num of instances for each class in the support set')
parser.add_argument('--Q', type=int, default=5, help='Num of instances for each class in the query set')
parser.add_argument('--batch', type=int, default=2, help='batch size')
parser.add_argument('--max_length', type=int, default=90, help='max length of sentence')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='initial learning rate')
parser.add_argument('--adv', default=None, help='adv file, None or pubmed_unsupervised.json')
parser.add_argument('--hidden_size', default=768, type=int, help='hidden size')
parser.add_argument('--roberta', default=False, type=bool, help='use roberta or not')
parser.add_argument('--cross_domain', default=False, type=bool, help='single domain or cross domain')
args = parser.parse_args()


print('setting:')
print(args)

print("{}-way(train)-{}-way(test)-{}-shot with batch {} Few-Shot Triple Extraction".format(args.N_for_train, args.N_for_test, args.K, args.Q))
print("Model: {}".format(args.model_name))

max_length = args.max_length

root_path = '/home/ycm/CODE/test/'
train_data_loader = JSONFileDataLoader(root_path + 'data/fewrel/train.json', max_length=max_length, reprocess=False, roberta=args.roberta)

if args.cross_domain:
    val_data_loader = JSONFileDataLoader('data/fewrel/new_val_pubmed.json', max_length=max_length, reprocess=False, is_same_domain=False, roberta=args.roberta)
    test_data_loader = JSONFileDataLoader('data/fewrel/new_test_pubmed.json', max_length=max_length, reprocess=False, is_same_domain=False, roberta=args.roberta)
else:
    val_data_loader = JSONFileDataLoader(root_path + 'data/fewrel/val.json', max_length=max_length, reprocess=False, is_same_domain=True, roberta=args.roberta)
    test_data_loader = JSONFileDataLoader(root_path + 'data/fewrel/test.json', max_length=max_length, reprocess=False, is_same_domain=True, roberta=args.roberta)


name = 'seed=' + str(seed) + '_MG-FRTE_prev'
log_file_path = 'record/' + name + '_'
model_name = args.model_name + "_" + name

model = MGFRTE(max_length, word_embedding_dim=args.hidden_size, args=args, roberta=args.roberta)

if args.adv:
    adv_data_loader = get_loader_unsupervised(args.adv, model,
                                                  N=args.N_for_train, K=args.K, Q=args.Q, batch_size=args.batch, root='data/')
    d = Discriminator(args.hidden_size)
    framework = FewShotREFramework(train_data_loader, val_data_loader, test_data_loader, log_file_path, name, adv_data_loader, adv=args.adv, roberta=args.roberta,
                                   d=d)
else:
    framework = FewShotREFramework(train_data_loader, val_data_loader, test_data_loader, log_file_path, name, roberta=args.roberta)


ckpt_dir = 'checkpoint'
    
framework.train(model, model_name, device, args.batch, N_for_train=args.N_for_train,  N_for_eval=args.N_for_test,
               K=args.K, Q=args.Q,  learning_rate=args.learning_rate, # pretrain_model=os.path.join(ckpt_dir, model_name + '.pth'),
               train_iter=10000, val_iter=500, val_step=1000, test_iter=1000, args=args)


if args.cross_domain:
    with torch.no_grad():
        ckpt_dir = 'checkpoint'
        test_acc = framework.eval(model, 1, 3, 3, 5, 1000, device, ckpt=os.path.join(ckpt_dir, model_name + '.pth'))
        print("{0:}-way-{1:}-shot test   Test accuracy: {2:3.2f}".format(3, 3, test_acc*100))

    with torch.no_grad():
        ckpt_dir = './checkpoint'
        test_acc = framework.eval(model, 1, 5, 5, 5, 1000, device, ckpt=os.path.join(ckpt_dir, model_name + '.pth'))
        print("{0:}-way-{1:}-shot test   Test accuracy: {2:3.2f}".format(5, 5, test_acc*100))
else:
    with torch.no_grad():
        ckpt_dir = './checkpoint'
        test_acc = framework.eval(model, 1, 5, 5, 5, 1000, device, ckpt=os.path.join(ckpt_dir, model_name + '.pth'))
        print("{0:}-way-{1:}-shot test   Test accuracy: {2:3.2f}".format(5, 5, test_acc*100))

    with torch.no_grad():
        ckpt_dir = './checkpoint'
        test_acc = framework.eval(model, 1, 10, 10, 5, 1000, device, ckpt=os.path.join(ckpt_dir, model_name + '.pth'))
        print("{0:}-way-{1:}-shot test   Test accuracy: {2:3.2f}".format(10, 10, test_acc*100))
