import torch
from model import ModelParam, MaskedKLDivLoss, MaskedNLLLoss
from util.write_file import WriteFile
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from tqdm import tqdm
from util.compare_to_save import compare_to_save
import test_process
import numpy as np

def dev_process(opt, critertion, cl_model, dev_loader, epoch, test_loader=None, last_F1=None, last_Accuracy=None, train_log=None, log_summary_writer=None, gamma_1=1.0):
    y_true = []
    y_pre = []
    total_labels = 0
    dev_loss = 0

    orgin_param = ModelParam()
    print("epoch", epoch)
    with torch.no_grad():
        cl_model.eval()
        dev_loader_tqdm = tqdm(dev_loader, desc='Dev Iteration')
        
        for index, data in enumerate(dev_loader_tqdm):
            texts_origin, bert_attention_mask, image_origin, text_image_mask, labels, \
                texts_augment, bert_attention_mask_augment, image_augment, text_image_mask_augment, target_labels, \
                raw_texts_batch, aspect_words_batch = data

            if opt.cuda is True:
                texts_origin = texts_origin.cuda()
                bert_attention_mask = bert_attention_mask.cuda()
                image_origin = image_origin.cuda()
                text_image_mask = text_image_mask.cuda()
                labels = labels.cuda()

            orgin_param.set_data_param(texts=texts_origin, bert_attention_mask=bert_attention_mask, images=image_origin,
                                       text_image_mask=text_image_mask, raw_texts=raw_texts_batch, aspect_words=aspect_words_batch)
            kl_loss = MaskedKLDivLoss()
            loss_function = MaskedNLLLoss()

            outputs_org, _ = cl_model(orgin_param, index, epoch, data_augment=None, labels=labels)
            (t_log_prob, i_log_prob, all_log_prob, all_prob, kl_t_log_prob, kl_a_log_prob, kl_all_prob) = outputs_org

            loss = gamma_1 * loss_function(all_log_prob, labels) + \
                   gamma_1 * (loss_function(t_log_prob, labels) + loss_function(i_log_prob, labels)) + \
                   gamma_1 * (kl_loss(kl_t_log_prob, kl_all_prob) + kl_loss(kl_a_log_prob, kl_all_prob))

            loss = loss / opt.acc_batch_size
            dev_loss += loss.item()
            
            _, predicted = torch.max(all_prob, 1)
            total_labels += labels.size(0)
            y_true.extend(labels.cpu())
            y_pre.extend(predicted.cpu())
            dev_loader_tqdm.set_description("Dev Iteration, loss: %.6f" % loss)

        dev_loss /= total_labels
        y_true = np.array(y_true)
        y_pre = np.array(y_pre)

        dev_accuracy = accuracy_score(y_true, y_pre)
        dev_F1_weighted = f1_score(y_true, y_pre, average='weighted')
        dev_R_weighted = recall_score(y_true, y_pre, average='weighted')
        dev_precision_weighted = precision_score(y_true, y_pre, average='weighted')
        dev_F1 = f1_score(y_true, y_pre, average='macro')
        dev_R = recall_score(y_true, y_pre, average='macro')
        dev_precision = precision_score(y_true, y_pre, average='macro')

        save_content = 'Dev  : Accuracy: %.6f, F1(weighted): %.6f, Precision(weighted): %.6f, R(weighted): %.6f, F1(macro): %.6f, Precision: %.6f, R: %.6f, loss: %.6f' % \
                       (dev_accuracy, dev_F1_weighted, dev_precision_weighted, dev_R_weighted, dev_F1, dev_precision, dev_R, dev_loss)
        print(save_content)

        if last_F1 is not None:
            WriteFile(opt.save_model_path, 'train_correct_log.txt', save_content + '\n', 'a+')
            test_process.test_process(opt, critertion, cl_model, test_loader, epoch, last_F1, log_summary_writer, train_log['epoch'])

            dev_log = {
                "dev_accuracy": dev_accuracy,
                "dev_F1": dev_F1,
                "dev_R": dev_R,
                "dev_precision": dev_precision,
                "dev_loss": dev_loss
            }

            last_Accuracy, is_save_model, model_name = compare_to_save(last_Accuracy, dev_accuracy, opt, cl_model, train_log, dev_log, 'Acc', opt.save_acc, add_enter=False)
            if is_save_model is True:
                if opt.data_type == 'HFM':
                    last_F1, is_save_model, model_name = compare_to_save(last_F1, dev_F1, opt, cl_model, train_log, dev_log, 'F1-marco', opt.save_F1, 'F1-marco', model_name)
                else:
                    last_F1, is_save_model, model_name = compare_to_save(last_F1, dev_F1_weighted, opt, cl_model, train_log, dev_log, 'F1', opt.save_F1, 'F1', model_name)
            else:
                if opt.data_type == 'HFM':
                    last_F1, is_save_model, model_name = compare_to_save(last_F1, dev_F1, opt, cl_model, train_log, dev_log, 'F1-marco', opt.save_F1)
                else:
                    last_F1, is_save_model, model_name = compare_to_save(last_F1, dev_F1_weighted, opt, cl_model, train_log, dev_log, 'F1', opt.save_F1)

            return last_F1, last_Accuracy