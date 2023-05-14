import os
import sys
import time
import logging
import torch
from torch import optim
import torch.nn as nn
from tqdm import tqdm
import openpyxl
from transformers import AdamW, get_linear_schedule_with_warmup


def writeToExcel(file_path, new_list):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = 'Predict Results'
    for r in range(len(new_list)):
        for c in range(len(new_list[0])):
            ws.cell(r + 1, c + 1).value = new_list[r][c]
    wb.save(file_path)


def create_logger(final_output_path):
    log_file = '{}.log'.format(time.strftime('%m-%d-%H-%M'))
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=final_output_path + log_file, format=head)
    clogger = logging.getLogger()
    clogger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    clogger.addHandler(ch)
    return clogger


class FewShotREFramework:

    def __init__(self, train_data_loader, val_data_loader, test_data_loader, log_file_path, model_name,
                 adv_data_loader=None, adv=False, roberta=False, d=None):
        '''
        train_data_loader: DataLoader for training.
        val_data_loader: DataLoader for validating.
        test_data_loader: DataLoader for testing.
        '''
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.record_dir = './record'
        self.result_dir = './result'
        if not os.path.exists(self.record_dir):
            os.makedirs(self.record_dir)
        self.logger = create_logger(log_file_path)
        self.model_name = model_name

        self.adv_data_loader = adv_data_loader
        self.roberta = roberta
        self.adv = adv
        if adv:
            self.adv_cost = nn.CrossEntropyLoss()
            self.d = d
            self.d.cuda()

    def __load_model__(self, ckpt):
        '''
        ckpt: Path of the checkpoint
        return: Checkpoint dict
        '''
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            print("Successfully loaded checkpoint '%s'" % ckpt)
            return checkpoint
        else:
            raise Exception("No checkpoint found at '%s'" % ckpt)

    def train(self, model, model_name, device, B=4, N_for_train=20, N_for_eval=5, K=5, Q=100,
              ckpt_dir='./checkpoint', learning_rate=1e-1, lr_step_size=20000,
              weight_decay=1e-5, train_iter=30000, val_iter=1000, val_step=2000,
              test_iter=3000, warmup_step=300, pretrain_model=None, args=None):
        '''
        model: model
        model_name: Name of the model
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        ckpt_dir: Directory of checkpoints
        test_result_dir: Directory of test results
        learning_rate: Initial learning rate
        lr_step_size: Decay learning rate every lr_step_size steps
        weight_decay: Rate of decaying weight
        train_iter: Num of iterations of training
        val_iter: Num of iterations of validating
        val_step: Validate every val_step steps
        test_iter: Num of iterations of testing
        cuda: Use CUDA or not
        pretrain_model: Pre-trained checkpoint path
        '''
        # Init
        parameters_to_optimize = filter(lambda x: x.requires_grad, model.parameters())
        bert_params = filter(lambda x: x.requires_grad, model.embedding.parameters())

        tmp_id = list(map(id, model.embedding.parameters()))
        other_params = filter(lambda x: x.requires_grad and id(x) not in tmp_id, parameters_to_optimize)
        optimizer = getattr(torch.optim, 'Adam')(
            [{'params': bert_params, 'lr': 1e-5},
             {'params': other_params, 'lr': learning_rate}],
            weight_decay=weight_decay
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size)
        
        if self.adv:
            optimizer_encoder = getattr(torch.optim, 'Adam')(
                    [{'params': bert_params, 'lr': 1e-5}],
                    weight_decay=weight_decay
            )
            optimizer_dis = optim.SGD(self.d.parameters(), lr=1e-2)

        self.logger.info(args)

        if pretrain_model:
            checkpoint = self.__load_model__(pretrain_model)
            model.load_state_dict(checkpoint['state_dict'])

        model = model.to(device)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        model = model.cuda()
        model.train()

        # Training
        best_acc = 0.0
        for it in range(0, train_iter):
            support, query, support_rel_label, query_rel_label, support_entity_label, query_entity_label, query_sent, \
            query_entity = self.train_data_loader.next_batch(device, B, N_for_train, K, Q)

            acc = self.eval(model, 1, N_for_eval, K, Q, val_iter, device, isTrain=False)
                    
            _, _, loss_entity, _, _, loss_relation = model(support, query, N_for_train, K, Q, support_rel_label,
                                                           query_rel_label, support_entity_label, query_entity_label,
                                                           device)
            loss = loss_entity + loss_relation

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Adv part
            if self.adv:
                support_adv = next(self.adv_data_loader)
                if torch.cuda.is_available():
                    for k in support_adv:
                        support_adv[k] = support_adv[k].cuda()

                features_ori = model.adv(support)
                features_adv = model.adv(support_adv)
                features = torch.cat([features_ori, features_adv], 0)
                total = features.size(0)
                dis_labels = torch.cat([torch.zeros((total // 2)).long().cuda(),
                                        torch.ones((total // 2)).long().cuda()], 0)
                dis_logits = self.d(features)
                loss_dis = self.adv_cost(dis_logits, dis_labels)
                _, pred = dis_logits.max(-1)
                right_dis = float((pred == dis_labels).long().sum()) / float(total)

                optimizer_dis.zero_grad()
                loss_dis.backward(retain_graph=True)
            
                loss_encoder = self.adv_cost(dis_logits, 1 - dis_labels)

                optimizer_encoder.zero_grad()
                loss_encoder.backward()

                optimizer_dis.step()
                optimizer_encoder.step()

            if it % 20 == 0:
                if self.adv:
                    self.logger.info(
                        'it:{}, model name:{}, loss:{:.5f}, entity_loss:{:.5f}, relation_loss:{:.5f}, adv_g_loss:{:.5f}, adv_d_loss:{:.5f}'.format(
                            it,
                            model_name,
                            loss.item(),
                            loss_entity.item(),
                            loss_relation.item(), loss_encoder.item(), loss_dis.item()))
                else:
                    self.logger.info(
                        'it:{}, model name:{}, loss:{:.5f}, entity_loss:{:.5f}, relation_loss:{:.5f}'.format(it,
                                                                                                             model_name,
                                                                                                             loss.item(),
                                                                                                             loss_entity.item(),
                                                                                                             loss_relation.item()
                                                                                                             ))

            if (it + 1) % val_step == 0:
                with torch.no_grad():
                    acc = self.eval(model, 1, N_for_eval, K, Q, val_iter, device, isTrain=False)
                    self.logger.info(
                        "{0:}---{1:}-way-{2:}-shot val   Val accuracy: {3:3.2f}".format(it, N_for_eval, K, acc * 100))

                    if acc > best_acc:
                        if not os.path.exists(ckpt_dir):
                            os.makedirs(ckpt_dir)
                        save_path = os.path.join(ckpt_dir, model_name + ".pth")
                        torch.save({'state_dict': model.state_dict()}, save_path)
                        best_acc = acc
                model.train()

        print("\n####################\n")
        print("Finish training " + model_name)
       

    def eval(self, model, B, N, K, Q, eval_iter, device, ckpt=None, isTrain=False):
        '''
        model: a FewShotREModel instance
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        eval_iter: Num of iterations
        ckpt: Checkpoint path. Set as None if using current model parameters.
        return: Accuracy
        '''
        print("testing:")
        if ckpt is None:
            eval_dataset = self.val_data_loader
        else:
            checkpoint = self.__load_model__(ckpt)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            model.cuda()
            eval_dataset = self.test_data_loader
        model.eval()

        rel_precision, rel_recall, rel_f1 = 0.0, 0.0, 0.0
        entity_precision, entity_recall, entity_f1 = 0.0, 0.0, 0.0
        triple_precision, triple_recall, triple_f1 = 0.0, 0.0, 0.0
        pack = [['sent', 'head_predict', 'head_gt', 'tail_predict', 'tail_gt']]
        for it in tqdm(range(eval_iter)):
            support, query, support_rel_label, query_rel_label, support_entity_label, query_entity_label, query_sent, \
            query_entity = eval_dataset.next_batch(device, B, N, K, Q)

            entity_predict, query_entity_label, _, rel_predict, rel_label, _ = model(support, query, N, K, Q,
                                                                                     support_rel_label, query_rel_label,
                                                                                     support_entity_label,
                                                                                     query_entity_label, device, isTrain=False)

            batch_entity_precision, batch_entity_recall, batch_entity_f1, rel_acc, batch_triple_precision, batch_triple_recall, batch_triple_f1, batch_pack = model.cal_f1(
                entity_predict, query_entity_label, query_sent, query_entity, rel_predict, rel_label)
                
            entity_precision += batch_entity_precision
            entity_recall += batch_entity_recall
            entity_f1 += batch_entity_f1
            rel_precision += rel_acc
            rel_recall += rel_acc
            rel_f1 += rel_acc
            triple_precision += batch_triple_precision
            triple_recall += batch_triple_recall
            triple_f1 += batch_triple_f1
            pack.extend(batch_pack)

        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        writeToExcel(self.result_dir + '/res_' + self.model_name + '.xlsx', pack)
        self.logger.info("Entity: precision: {:.4f}, recall: {:.4f}, f1: {:.4f}".format(entity_precision / eval_iter,
                                                                                        entity_recall / eval_iter,
                                                                                        entity_f1 / eval_iter))
        self.logger.info("Relation: precision: {:.4f}, recall: {:.4f}, f1: {:.4f}".format(rel_precision / eval_iter,
                                                                                          rel_recall / eval_iter,
                                                                                          rel_f1 / eval_iter))
        self.logger.info("Triple: precision: {:.4f}, recall: {:.4f}, f1: {:.4f}".format(triple_precision / eval_iter,
                                                                                        triple_recall / eval_iter,
                                                                                        triple_f1 / eval_iter))
        return triple_f1 / eval_iter
