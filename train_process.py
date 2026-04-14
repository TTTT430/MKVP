import torch
from torch.optim import Adam, AdamW, SGD
from tqdm import tqdm, trange
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from util.write_file import WriteFile
import dev_process
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from model import ModelParam,MaskedKLDivLoss,MaskedNLLLoss
import torch.nn.modules as nn
import matplotlib.pyplot as plt
import os


class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1.0, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def train_process(opt, train_loader, dev_loader, test_loader, cl_model, critertion,
                  log_summary_writer: SummaryWriter = None, gamma_1=1.0, gamma_2=1.0, gamma_3=1.0, tokenizer=None,
                  image_id_list=None):
    optimizer = None

    pre_train_model_param = [name for name, param in cl_model.named_parameters() if 'text_model' in name]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in cl_model.named_parameters() if n in pre_train_model_param], "lr": 0},
        {"params": [p for n, p in cl_model.named_parameters() if n not in pre_train_model_param], "lr": opt.fuse_lr},
    ]

    if opt.optim == 'adam':
        optimizer = Adam(optimizer_grouped_parameters, betas=(opt.optim_b1, opt.optim_b2))
    elif opt.optim == 'adamw':
        optimizer = AdamW(optimizer_grouped_parameters, betas=(opt.optim_b1, opt.optim_b2), weight_decay=0.05)
    elif opt.optim == 'sgd':
        optimizer = SGD(optimizer_grouped_parameters, momentum=opt.momentum)

    orgin_param = ModelParam()
    augment_param = ModelParam()

    history = {
        'epoch': [], 'train_acc': [], 'dev_acc': []
    }

    last_F1 = 0
    last_Accuracy = 0

    fgm = FGM(cl_model)
    ema = EMA(cl_model, 0.999)
    ema.register()

    for epoch in trange(opt.epoch, desc='Epoch:'):
        y_true = []
        y_pre = []
        run_loss = 0
        total_labels = 0
        new_epoch = epoch

        cl_model.train()
        cl_model.zero_grad()

        if epoch >= opt.train_fuse_model_epoch:
            decay_rate = 0.5 ** ((epoch - opt.train_fuse_model_epoch) // 10)
            current_lr = opt.lr * decay_rate
            optimizer.param_groups[0]['lr'] = current_lr
            optimizer.param_groups[1]['lr'] = current_lr

        train_loader_tqdm = tqdm(train_loader, desc='Train Iteration:')
        epoch_step_num = epoch * train_loader_tqdm.total
        step_num = 0

        for index, data in enumerate(train_loader_tqdm):
            texts_origin, bert_attention_mask, image_origin, text_image_mask, labels, \
                texts_augment, bert_attention_mask_augment, image_augment, text_image_mask_augment, target_labels, \
                raw_texts_batch, aspect_words_batch = data

            if opt.cuda is True:
                texts_origin = texts_origin.cuda()
                bert_attention_mask = bert_attention_mask.cuda()
                image_origin = image_origin.cuda()
                text_image_mask = text_image_mask.cuda()
                labels = labels.cuda()
                texts_augment = texts_augment.cuda()
                bert_attention_mask_augment = bert_attention_mask_augment.cuda()
                image_augment = image_augment.cuda()
                text_image_mask_augment = text_image_mask_augment.cuda()
                for i in range(len(target_labels)):
                    target_labels[i] = target_labels[i].cuda()

            orgin_param.set_data_param(texts=texts_origin, bert_attention_mask=bert_attention_mask, images=image_origin,
                                       text_image_mask=text_image_mask, raw_texts=raw_texts_batch,
                                       aspect_words=aspect_words_batch)
            augment_param.set_data_param(texts=texts_augment, bert_attention_mask=bert_attention_mask_augment,
                                         images=image_augment, text_image_mask=text_image_mask_augment)

            kl_loss = MaskedKLDivLoss()
            loss_function = MaskedNLLLoss()

            outputs_org, outputs_aug = cl_model(data_orgin=orgin_param, index=index, epoch=new_epoch,
                                                data_augment=augment_param, labels=labels)
            (t_log_prob, i_log_prob, all_log_prob, all_prob, kl_t_log_prob, kl_a_log_prob, kl_all_prob) = outputs_org

            main_loss = opt.lambda_ce * loss_function(all_log_prob, labels) + \
                        opt.lambda_ce * (loss_function(t_log_prob, labels) + loss_function(i_log_prob, labels)) + \
                        opt.lambda_ce * (kl_loss(kl_t_log_prob, kl_all_prob) + kl_loss(kl_a_log_prob, kl_all_prob))

            contrastive_loss = 0.0
            if outputs_aug is not None:
                (aug_t_log_prob, aug_i_log_prob, aug_all_log_prob, aug_all_prob, aug_kl_t_log_prob, aug_kl_a_log_prob,
                 aug_kl_all_prob) = outputs_aug
                cl_loss_1 = kl_loss(all_log_prob, aug_all_prob)
                cl_loss_2 = kl_loss(aug_all_log_prob, all_prob)
                contrastive_loss = opt.lambda_kl * 10 * (cl_loss_1 + cl_loss_2) / 2

            loss = main_loss + contrastive_loss
            loss = loss / opt.acc_batch_size
            loss.backward()

            fgm.attack(epsilon=1.0, emb_name='word_embeddings')
            outputs_org_adv, outputs_aug_adv = cl_model(data_orgin=orgin_param, index=index, epoch=new_epoch,
                                                        data_augment=augment_param, labels=labels)
            (t_log_prob_adv, i_log_prob_adv, all_log_prob_adv, all_prob_adv, _, _, _) = outputs_org_adv

            loss_adv = opt.lambda_adv * loss_function(all_log_prob_adv, labels) + \
                       opt.lambda_adv * (loss_function(t_log_prob_adv, labels) + loss_function(i_log_prob_adv, labels))

            loss_adv = loss_adv / opt.acc_batch_size
            loss_adv.backward()

            fgm.restore(emb_name='word_embeddings')

            train_loader_tqdm.set_description(
                "Train Iteration, loss: %.6f, lr: %e" % (loss, optimizer.param_groups[0]['lr']))

            if (index + 1) % opt.acc_grad == 0:
                if log_summary_writer:
                    log_summary_writer.add_scalar('train_info/loss', loss.item(), global_step=step_num + epoch_step_num)
                optimizer.step()
                ema.update()
                optimizer.zero_grad()
            step_num += 1

            _, predicted = torch.max(all_prob, 1)
            y_true.extend(labels.cpu())
            y_pre.extend(predicted.cpu())
            run_loss += loss.item()
            total_labels += labels.size(0)

        run_loss /= total_labels
        y_true = np.array(y_true)
        y_pre = np.array(y_pre)

        train_accuracy = accuracy_score(y_true, y_pre)
        train_F1_weighted = f1_score(y_true, y_pre, average='weighted')
        train_R_weighted = recall_score(y_true, y_pre, average='weighted')
        train_precision_weighted = precision_score(y_true, y_pre, average='weighted')
        train_F1 = f1_score(y_true, y_pre, average='macro')
        train_R = recall_score(y_true, y_pre, average='macro')
        train_precision = precision_score(y_true, y_pre, average='macro')

        save_content = 'Epoch: %d:\nTrain: Accuracy: %.6f, F1(weighted): %.6f, Precision(weighted): %.6f, R(weighted): %.6f, F1(macro): %.6f, Precision: %.6f, R: %.6f, loss: %.6f' % \
                       (epoch, train_accuracy, train_F1_weighted, train_precision_weighted, train_R_weighted, train_F1,
                        train_precision, train_R, run_loss)
        WriteFile(opt.save_model_path, 'train_correct_log.txt', save_content + '\n', 'a+')
        print(save_content, ' ' * 100)

        train_log = {
            "epoch": epoch,
            "train_accuracy": train_accuracy,
            "train_F1": train_F1,
            "run_loss": run_loss
        }

        ema.apply_shadow()
        last_F1, last_Accuracy = dev_process.dev_process(
            opt, critertion, cl_model, dev_loader, epoch, test_loader, last_F1, last_Accuracy, train_log,
            log_summary_writer
        )
        ema.restore()

        # 绘制仅保留模型实际准确率的图表
        history['epoch'].append(epoch)
        history['train_acc'].append(train_accuracy)
        history['dev_acc'].append(last_Accuracy)

        plt.figure(figsize=(8, 5))
        plt.plot(history['epoch'], history['train_acc'], label='Train Accuracy', marker='.')
        plt.plot(history['epoch'], history['dev_acc'], label='Validation Accuracy', marker='.')
        plt.title('MKVP-Net Training & Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(opt.save_model_path, 'accuracy_curves.png'))
        plt.close()
