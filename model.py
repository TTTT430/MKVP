import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as cv_models
import os
import math
import copy
import spacy
from senticnet.senticnet import SenticNet
from transformers import BertConfig, BertForPreTraining, AutoModel, AutoConfig, BertTokenizer, AutoTokenizer
from pre_model import RobertaEncoder

class MaskedKLDivLoss(nn.Module):
    def __init__(self):
        super(MaskedKLDivLoss, self).__init__()
        self.loss = nn.KLDivLoss(reduction='sum')

    def forward(self, log_pred, target):
        loss = self.loss(log_pred, target)
        return loss

class MaskedNLLLoss(nn.Module):
    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight, reduction='sum')

    def forward(self, pred, target):
        if type(self.weight) == type(None):
            loss = self.loss(pred , target)
        else:
            loss = self.loss(pred, target) / torch.sum(self.weight[target])
        return loss

class ModelParam:
    def __init__(self, texts=None, images=None, bert_attention_mask=None, text_image_mask=None, segment_token=None, image_coordinate_position_token=None, raw_texts=None, aspect_words=None):
        self.set_data_param(texts, images, bert_attention_mask, text_image_mask, segment_token, image_coordinate_position_token, raw_texts, aspect_words)

    def set_data_param(self, texts=None, images=None, bert_attention_mask=None, text_image_mask=None, segment_token=None, image_coordinate_position_token=None, raw_texts=None, aspect_words=None):
        self.texts = texts
        self.images = images
        self.bert_attention_mask = bert_attention_mask
        self.text_image_mask = text_image_mask
        self.segment_token = segment_token
        self.image_coordinate_position_token = image_coordinate_position_token
        self.raw_texts = raw_texts
        self.aspect_words = aspect_words

def get_extended_attention_mask(attention_mask, input_shape):
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})")

    extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask

class ActivateFun(nn.Module):
    def __init__(self, opt):
        super(ActivateFun, self).__init__()
        self.activate_fun = opt.activate_fun

    def _gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def forward(self, x):
        if self.activate_fun == 'relu':
            return torch.relu(x)
        elif self.activate_fun == 'gelu':
            return self._gelu(x)

class TextModel(nn.Module):
    def __init__(self, opt):
        super(TextModel, self).__init__()
        abl_path = ''

        if opt.text_model == 'bert-base':
            model_dir = os.path.join(abl_path, "bert-base-uncased")
            self.config = BertConfig.from_pretrained(model_dir)
            self.model = BertForPreTraining.from_pretrained(model_dir, config=self.config)
            self.model = self.model.bert
        elif opt.text_model == 'bertweet':
            model_dir = os.path.join(abl_path, "bertweet-base")
            print(f"Loading BERTweet model from local: {model_dir}")
            self.config = AutoConfig.from_pretrained(model_dir)
            self.model = AutoModel.from_pretrained(model_dir, config=self.config)

        for param in self.model.parameters():
            param.requires_grad = True

        if hasattr(self.config, 'hidden_size'):
            self.output_dim = self.config.hidden_size
        else:
            self.output_dim = 768

    def get_output_dim(self):
        return self.output_dim

    def get_config(self):
        return self.config

    def get_encoder(self):
        model_encoder = copy.deepcopy(self.model.encoder)
        return model_encoder

    def forward(self, input, attention_mask):
        token_type_ids = torch.zeros_like(input)
        output = self.model(input, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return output.last_hidden_state, output.pooler_output

class ImageModel(nn.Module):
    def __init__(self, opt):
        super(ImageModel, self).__init__()
        if opt.image_model == 'resnet-152':
            self.resnet = cv_models.resnet152(pretrained=True)
        elif opt.image_model == 'resnet-101':
            self.resnet = cv_models.resnet101(pretrained=True)
        elif opt.image_model == 'resnet-50':
            self.resnet = cv_models.resnet50(pretrained=True)
        elif opt.image_model == 'resnet-34':
            self.resnet = cv_models.resnet34(pretrained=True)
        elif opt.image_model == 'resnet-18':
            self.resnet = cv_models.resnet18(pretrained=True)

        self.resnet_encoder = nn.Sequential(*(list(self.resnet.children())[:-2]))
        self.resnet_avgpool = nn.Sequential(list(self.resnet.children())[-2])
        self.output_dim = self.resnet_encoder[7][2].conv3.out_channels

        for param in self.resnet.parameters():
            if opt.fixed_image_model:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def get_output_dim(self):
        return self.output_dim

    def forward(self, images):
        image_encoder = self.resnet_encoder(images)
        image_cls = self.resnet_avgpool(image_encoder)
        image_cls = torch.flatten(image_cls, 1)
        return image_encoder, image_cls

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class KnowledgeGuidanceLayer(nn.Module):
    def __init__(self, hidden_dim, dropout=0.3):
        super(KnowledgeGuidanceLayer, self).__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, H_in, knowledge_guidance_matrix):
        support = self.W(H_in)
        output = torch.bmm(knowledge_guidance_matrix, support)
        H_out = self.relu(output)
        H_out = self.dropout(H_out)
        return H_out

class VCP_Module(nn.Module):
    def __init__(self, opt, hidden_dim, num_layers=2):
        super(VCP_Module, self).__init__()
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.sn = SenticNet()
            abl_path = ''

            if opt.text_model == 'bertweet':
                model_dir = os.path.join(abl_path, "bertweet-base")
                print(f"VCP Loading Tokenizer from: {model_dir}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_dir, normalization=True, use_fast=False)
            else:
                model_dir = os.path.join(abl_path, "bert-base-uncased")
                print(f"VCP Loading Tokenizer from: {model_dir}")
                self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        except Exception as e:
            print(f"Error initializing spacy, SenticNet or Tokenizer: {e}")
            self.nlp = None
            self.sn = None
            self.tokenizer = None

        self.num_layers = num_layers
        self.knowledge_layers = nn.ModuleList([KnowledgeGuidanceLayer(hidden_dim) for _ in range(num_layers)])

    def forward(self, text_features, raw_texts, aspect_words_batch):
        device = text_features.device
        bert_seq_len = text_features.size(1)
        adj_matrices = []

        if self.nlp is None or self.sn is None or self.tokenizer is None:
            return text_features

        for i, text in enumerate(raw_texts):
            if not isinstance(text, str):
                adj = np.eye(bert_seq_len, dtype=np.float32)
                adj_matrices.append(torch.from_numpy(adj))
                continue

            doc = self.nlp(text)
            words = [token.text for token in doc]
            n_spacy = len(words)

            spacy_adj = np.eye(n_spacy, dtype=np.float32)
            for token in doc:
                for child in token.children:
                    if token.i < n_spacy and child.i < n_spacy:
                        spacy_adj[token.i, child.i] = 1
                        spacy_adj[child.i, token.i] = 1

            sentiment_matrix = np.zeros((n_spacy, n_spacy), dtype=np.float32)
            word_sentiments = []
            for word in words:
                try:
                    val = float(self.sn.concept(word)['polarity_value'])
                except:
                    val = 0.0
                word_sentiments.append(val)

            for r in range(n_spacy):
                for c in range(n_spacy):
                    sentiment_matrix[r, c] = word_sentiments[r] + word_sentiments[c]

            aspect_matrix = np.zeros((n_spacy, n_spacy), dtype=np.float32)
            aspects = aspect_words_batch[i] if aspect_words_batch is not None and i < len(aspect_words_batch) else []
            for r in range(n_spacy):
                for c in range(n_spacy):
                    if words[r] in aspects or words[c] in aspects:
                        aspect_matrix[r, c] = 1

            multi_source_knowledge_matrix = spacy_adj * (np.abs(sentiment_matrix) + aspect_matrix + 1.0)

            bert_adj = np.zeros((bert_seq_len, bert_seq_len), dtype=np.float32)
            bert_adj[0, 0] = 1.0

            current_bert_idx = 1
            spacy_to_bert_map = {}

            for spacy_idx, spacy_token in enumerate(doc):
                sub_tokens = self.tokenizer.tokenize(spacy_token.text)
                num_sub_tokens = max(len(sub_tokens), 1)
                end_bert_idx = current_bert_idx + num_sub_tokens

                if current_bert_idx >= bert_seq_len:
                    spacy_to_bert_map[spacy_idx] = []
                else:
                    real_end = min(end_bert_idx, bert_seq_len)
                    spacy_to_bert_map[spacy_idx] = list(range(current_bert_idx, real_end))

                current_bert_idx = end_bert_idx

            for r in range(n_spacy):
                for c in range(n_spacy):
                    weight = multi_source_knowledge_matrix[r, c]
                    if weight != 0:
                        bert_rows = spacy_to_bert_map.get(r, [])
                        bert_cols = spacy_to_bert_map.get(c, [])
                        for br in bert_rows:
                            for bc in bert_cols:
                                bert_adj[br, bc] = weight

            for k in range(current_bert_idx, bert_seq_len):
                bert_adj[k, k] = 1.0

            row_sum = bert_adj.sum(axis=1)
            row_sum[row_sum == 0] = 1e-6
            bert_adj_norm = bert_adj / row_sum[:, np.newaxis]
            adj_matrices.append(torch.from_numpy(bert_adj_norm))

        knowledge_guidance_tensor = torch.stack(adj_matrices, dim=0).to(device)

        H = text_features
        for layer in self.knowledge_layers:
            H = layer(H, knowledge_guidance_tensor)

        return H

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.3):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.actv = gelu
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.actv(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x

class MultiHeadedAttention(nn.Module):
    def __init__(self, head_count, model_dim, dropout=0.3):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_k = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_v = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_q = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(model_dim, model_dim)

    def forward(self, key, value, query):
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        key = self.linear_k(key).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)
        value = self.linear_v(value).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)
        query = self.linear_q(query).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        query = query / math.sqrt(dim_per_head)
        scores = torch.matmul(query, key.transpose(2, 3))
        attn = self.softmax(scores)
        drop_attn = self.dropout(attn)
        context = torch.matmul(drop_attn, value).transpose(1, 2).contiguous().view(batch_size, -1, head_count * dim_per_head)
        output = self.linear(context)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        L = x.size(1)
        pos_emb = self.pe[:, :L]
        x = x + pos_emb
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, opt, d_model, heads, d_ff, dropout,):
        super(TransformerEncoderLayer, self).__init__()
        self.opt = opt
        self.self_attn = MultiHeadedAttention(heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, inputs_a, inputs_b):
        if inputs_a.equal(inputs_b):
            if iter != 0:
                inputs_b = self.layer_norm(inputs_b)
            context = self.self_attn(inputs_b, inputs_b, inputs_b)
            out = self.dropout(context) + inputs_b
        else:
            if iter != 0:
                inputs_b = self.layer_norm(inputs_b)
            context = self.self_attn(inputs_a, inputs_a, inputs_b)

            if context.size(1) != inputs_a.size(1):
                context = context.permute(0, 2, 1)
                context = nn.AdaptiveAvgPool1d(inputs_a.size(1))(context)
                context = context.permute(0, 2, 1)

            out = self.dropout(context) + inputs_a

        return self.feed_forward(out)

class Multimodal_GatedFusion(nn.Module):
    def __init__(self, hidden_size):
        super(Multimodal_GatedFusion, self).__init__()
        self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, a, b):
        a_new = a.unsqueeze(-2)
        b_new = b.unsqueeze(-2)
        utters = torch.cat([a_new, b_new], dim=-2)
        utters_fc = torch.cat([self.fc(a).unsqueeze(-2), self.fc(b).unsqueeze(-2)], dim=-2)
        utters_softmax = self.softmax(utters_fc)
        utters_three_model = utters_softmax * utters
        final_rep = torch.sum(utters_three_model, dim=-2, keepdim=False)
        return final_rep

class TransformerEncoder(nn.Module):
    def __init__(self, opt, d_model, d_ff, heads, layers, dropout=0.1,):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.layers = layers
        self.pos_emb = PositionalEncoding(d_model)
        self.transformer_inter = nn.ModuleList([TransformerEncoderLayer(opt, d_model, heads, d_ff, dropout) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_a, x_b):
        if x_a.equal(x_b):
            x_b = self.pos_emb(x_b)
            x_b = self.dropout(x_b)
            for i in range(self.layers):
                x_b = self.transformer_inter[i](i, x_b, x_b)
        else:
            x_a = self.pos_emb(x_a)
            x_a = self.dropout(x_a)
            x_b = self.pos_emb(x_b)
            x_b = self.dropout(x_b)
            for i in range(self.layers):
                x_b = self.transformer_inter[i](i, x_a, x_b)
        return x_b

class Transformer_Based_Model(nn.Module):
    def __init__(self, opt, temp=1, n_head=8, n_classes=3, hidden_dim=768, dropout=0.2, image_output_type="all"):
        super(Transformer_Based_Model, self).__init__()
        self.temp = temp
        self.opt = opt
        self.n_classes = n_classes
        self.image_output_type = image_output_type

        self.text_model = TextModel(opt)
        self.image_model = ImageModel(opt)
        self.fuse_type = opt.fuse_type
        self.tran_dim = opt.tran_dim

        self.image_config = copy.deepcopy(self.text_model.get_config())
        self.text_config = copy.deepcopy(self.text_model.get_config())

        self.image_config.num_attention_heads = opt.tran_dim // 64
        self.image_config.hidden_size = opt.tran_dim
        self.image_config.num_hidden_layers = opt.image_num_layers
        self.image_encoder = RobertaEncoder(self.image_config)

        self.text_change = nn.Sequential(
            nn.Linear(self.text_model.get_output_dim(), self.tran_dim),
            ActivateFun(opt)
        )
        self.image_change = nn.Sequential(
            nn.Linear(self.image_model.get_output_dim(), self.tran_dim),
            ActivateFun(opt)
        )
        self.image_cls_change = nn.Sequential(
            nn.Linear(self.image_model.get_output_dim(), self.tran_dim),
            ActivateFun(opt)
        )

        self.t_t = TransformerEncoder(opt, d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout,)
        self.i_t = TransformerEncoder(opt, d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)

        self.i_i = TransformerEncoder(opt, d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.t_i = TransformerEncoder(opt, d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)

        self.vcp_module = VCP_Module(opt, hidden_dim=hidden_dim, num_layers=2)

        self.last_gate = Multimodal_GatedFusion(hidden_dim)

        self.t_output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )
        self.a_output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )

        self.all_output_layer = nn.Linear(hidden_dim, n_classes)

    def forward(self, text_inputs, image_inputs, text_image_mask, index, epoch, attention_mask=None, raw_texts=None, aspect_words=None):
        last_hidden_state, pooler_output = self.text_model(text_inputs, attention_mask=attention_mask)
        textf = self.text_change(last_hidden_state)


        if raw_texts is not None and aspect_words is not None:
            textf_vcp = self.vcp_module(last_hidden_state, raw_texts, aspect_words)
        else:
            device = textf.device
            textf_vcp = torch.zeros_like(textf).to(device)

        textf_for_t_i = textf + textf_vcp

        icouf, image_cls = self.image_model(image_inputs)
        if self.image_output_type == 'all':
            image_encoder = icouf.contiguous().view(icouf.size(0), -1, icouf.size(1))
            image_encoder_init = self.image_change(image_encoder)
            image_cls_init = self.image_cls_change(image_cls)
            icouf = torch.cat((image_cls_init.unsqueeze(1), image_encoder_init), dim=1)
        else:
            image_cls_init = self.image_cls_change(image_cls)
            icouf = image_cls_init.unsqueeze(1)

        image_mask = text_image_mask[:, -icouf.size(1):]
        extended_attention_mask = get_extended_attention_mask(image_mask, icouf.size())
        image_init = self.image_encoder(icouf,
                                        encoder_attention_mask=extended_attention_mask,
                                        output_attentions=self.text_config.output_attentions,
                                        output_hidden_states=self.text_config.output_hidden_states,
                                        return_dict=self.text_config.use_return_dict)
        icouf = image_init.last_hidden_state

        t_t_transformer_out = self.t_t(textf, textf)
        i_i_transformer_out = self.i_i(icouf, icouf)

        i_t_transformer_out = self.i_t(icouf, textf)
        t_i_transformer_out = self.t_i(textf_for_t_i, icouf)

        t_transformer_out_mix = torch.cat([t_t_transformer_out, i_t_transformer_out], dim=1)
        i_transformer_out_mix = torch.cat([i_i_transformer_out, t_i_transformer_out], dim=1)

        all_transformer_out = self.last_gate(t_transformer_out_mix, i_transformer_out_mix)

        text_dim = t_transformer_out_mix.size(2)
        t_pooled = torch.sum(t_transformer_out_mix, dim=1) / text_dim
        image_dim = i_i_transformer_out.size(2)
        i_pooled = torch.sum(i_i_transformer_out, dim=1) / image_dim

        if self.fuse_type == 'max':
            all_transformer_out = torch.max(all_transformer_out, dim=1)[0]
        elif self.fuse_type == 'ave':
            fusion_dim = all_transformer_out.size(2)
            all_transformer_out = torch.sum(all_transformer_out, dim=1) / fusion_dim
        else:
            raise Exception('fuse_type设定错误')

        t_final_out = self.t_output_layer(t_pooled)
        i_final_out = self.a_output_layer(i_pooled)
        all_final_out = self.all_output_layer(all_transformer_out)

        t_log_prob = F.log_softmax(t_final_out, 1)
        i_log_prob = F.log_softmax(i_final_out, 1)
        all_log_prob = F.log_softmax(all_final_out, 1)
        all_prob = F.softmax(all_final_out, 1)

        kl_t_log_prob = F.log_softmax(t_final_out / self.temp, 1)
        kl_a_log_prob = F.log_softmax(i_final_out / self.temp, 1)
        kl_all_prob = F.softmax(all_final_out / self.temp, 1)

        return t_log_prob, i_log_prob, all_log_prob, all_prob, kl_t_log_prob, kl_a_log_prob, kl_all_prob

class CLModel(nn.Module):
    def __init__(self, opt):
        super(CLModel, self).__init__()
        self.Transformer_Based_Model = Transformer_Based_Model(opt)

    def forward(self, data_orgin: ModelParam, index, epoch, data_augment: ModelParam = None, labels=None, target_labels=None):
        org_outputs = self.Transformer_Based_Model(
            text_inputs=data_orgin.texts,
            image_inputs=data_orgin.images,
            text_image_mask=data_orgin.text_image_mask,
            index=index,
            epoch=epoch,
            attention_mask=data_orgin.bert_attention_mask,
            raw_texts=data_orgin.raw_texts,
            aspect_words=data_orgin.aspect_words
        )

        if data_augment is not None and data_augment.texts is not None:
            aug_outputs = self.Transformer_Based_Model(
                text_inputs=data_augment.texts,
                image_inputs=data_augment.images,
                text_image_mask=data_augment.text_image_mask,
                index=index,
                epoch=epoch,
                attention_mask=data_augment.bert_attention_mask,
                raw_texts=None,
                aspect_words=None
            )
            return org_outputs, aug_outputs
        else:
            return org_outputs, None
