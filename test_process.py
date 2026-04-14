import torch
from model import ModelParam, MaskedKLDivLoss, MaskedNLLLoss
from util.write_file import WriteFile
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from tqdm import tqdm
import numpy as np

def test_process(opt, critertion, cl_model, test_loader, epoch, last_F1=None, log_summary_writer=None, gamma_1=1.0):
    y_true = []
    y_pre = []
    total_labels = 0
    test_loss = 0

    orgin_param = ModelParam()

    with torch.no_grad():
        cl_model.eval()
        test_loader_tqdm = tqdm(test_loader, desc='Test Iteration')

        for index, data in enumerate(test_loader_tqdm):
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
            test_loss += loss.item()
            
            _, predicted = torch.max(all_prob, 1)
            total_labels += labels.size(0)
            y_true.extend(labels.cpu())
            y_pre.extend(predicted.cpu())

            test_loader_tqdm.set_description("Test Iteration, loss: %.6f" % loss)

        test_loss /= total_labels
        y_true = np.array(y_true)
        y_pre = np.array(y_pre)

        test_accuracy = accuracy_score(y_true, y_pre)
        test_F1 = f1_score(y_true, y_pre, average='macro')
        test_R = recall_score(y_true, y_pre, average='macro')
        test_precision = precision_score(y_true, y_pre, average='macro')
        test_F1_weighted = f1_score(y_true, y_pre, average='weighted')
        test_R_weighted = recall_score(y_true, y_pre, average='weighted')
        test_precision_weighted = precision_score(y_true, y_pre, average='weighted')

        save_content = 'Test : Accuracy: %.6f, F1(weighted): %.6f, Precision(weighted): %.6f, R(weighted): %.6f, F1(macro): %.6f, Precision: %.6f, R: %.6f, loss: %.6f' % \
            (test_accuracy, test_F1_weighted, test_precision_weighted, test_R_weighted, test_F1, test_precision, test_R, test_loss)

        print(save_content)

        if last_F1 is not None:
            WriteFile(opt.save_model_path, 'train_correct_log.txt', save_content + '\n', 'a+')

        return test_accuracy, test_F1