import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import LSCLinear
from visual_net import resnet18


class QstEncoder(nn.Module):

    def __init__(self, qst_vocab_size, word_embed_size=512, embed_size=512, num_layers=1, hidden_size=512):
        super(QstEncoder, self).__init__()
        self.word2vec = nn.Embedding(qst_vocab_size, word_embed_size)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(word_embed_size, hidden_size, num_layers)
        self.fc = nn.Linear(2 * num_layers * hidden_size, embed_size)  # 2 for hidden and cell states

    def forward(self, question):
        qst_vec = self.word2vec(question)  # [batch_size, max_qst_length=30, word_embed_size=300]
        qst_vec = self.tanh(qst_vec)
        qst_vec = qst_vec.transpose(0, 1)  # [max_qst_length=30, batch_size, word_embed_size=300]
        self.lstm.flatten_parameters()
        _, (hidden, cell) = self.lstm(qst_vec)  # [num_layers=2, batch_size, hidden_size=512]
        qst_feature = torch.cat((hidden, cell), 2)  # [num_layers=2, batch_size, 2*hidden_size=1024]
        qst_feature = qst_feature.transpose(0, 1)  # [batch_size, num_layers=2, 2*hidden_size=1024]
        qst_feature = qst_feature.reshape(qst_feature.size()[0], -1)  # [batch_size, 2*num_layers*hidden_size=2048]
        qst_feature = self.tanh(qst_feature)
        qst_feature = self.fc(qst_feature)  # [batch_size, embed_size]

        return qst_feature

    def incremental_vocab(self, vocab_size):
        new_word2vec = nn.Embedding(vocab_size, 512).to(self.word2vec.weight.device)
        new_word2vec.weight.data[:self.word2vec.weight.size(0)] = self.word2vec.weight.data
        nn.init.kaiming_uniform_(new_word2vec.weight.data[self.word2vec.weight.size(0):])
        self.word2vec = new_word2vec

class IncreAudioVisualNet(nn.Module):
    def __init__(self, args, step_out_ans_num, vocab_size, LSC=False):
        super(IncreAudioVisualNet, self).__init__()
        self.args = args
        self.modality = args.modality
        self.vocab_size = vocab_size
        self.num_ans = step_out_ans_num

        if self.modality != 'visual' and self.modality != 'audio' and self.modality != 'audio-visual':
            raise ValueError('modality must be \'visual\', \'audio\' or \'audio-visual\'')
        if self.modality == 'visual':
            self.visual_proj = nn.Linear(768, 768)
        elif self.modality == 'audio':
            self.audio_proj = nn.Linear(768, 768)
        else:
            self.fc_a1 = nn.Linear(128, 512)
            self.fc_a2 = nn.Linear(512, 512)

            self.visual_net = resnet18(pretrained=True)

            self.fc_v = nn.Linear(2048, 512)
            self.fc_st = nn.Linear(512, 512)
            self.fc_fusion = nn.Linear(1024, 512)
            self.fc = nn.Linear(1024, 512)
            self.fc_aq = nn.Linear(512, 512)
            self.fc_vq = nn.Linear(512, 512)

            self.linear11 = nn.Linear(512, 512)
            self.dropout1 = nn.Dropout(0.1)
            self.linear12 = nn.Linear(512, 512)

            self.linear21 = nn.Linear(512, 512)
            self.dropout2 = nn.Dropout(0.1)
            self.linear22 = nn.Linear(512, 512)
            self.norm1 = nn.LayerNorm(512)
            self.norm2 = nn.LayerNorm(512)
            self.dropout3 = nn.Dropout(0.1)
            self.dropout4 = nn.Dropout(0.1)
            self.norm3 = nn.LayerNorm(512)

            self.attn_a = nn.MultiheadAttention(512, 4, dropout=0.1)
            self.attn_v = nn.MultiheadAttention(512, 4, dropout=0.1)
            self.question_encoder = QstEncoder(self.vocab_size)

            self.tanh = nn.Tanh()
            self.dropout = nn.Dropout(0.5)
            self.fc_ans = nn.Linear(512, self.num_ans)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc_gl = nn.Linear(1024, 512)

            # combine
            self.fc1 = nn.Linear(1024, 512)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(512, 256)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(256, 128)
            self.relu3 = nn.ReLU()
            self.fc4 = nn.Linear(128, 2)
            self.relu4 = nn.ReLU()

            self.temporal_conv = nn.Conv1d(512, 512, kernel_size=3, padding=1)

        if LSC:
            self.classifier = LSCLinear(512, self.num_ans)
        else:
            self.classifier = self.fc_ans

    def forward(self, audio, visual, question, out_logits=True, out_features=False, out_features_norm=False, out_feature_before_fusion=False, que_feature=False):
        if self.modality == 'visual':
            if visual is None:
                raise ValueError('input frames are None when modality contains visual')
            visual_feature = torch.mean(visual, dim=1)
            visual_feature = F.relu(self.visual_proj(visual_feature))
            logits = self.classifier(visual_feature)
        elif self.modality == 'audio':
            if audio is None:
                raise ValueError('input audio are None when modality contains audio')
            audio_feature = F.relu(self.audio_proj(audio))
            logits = self.classifier(audio_feature)
            outputs = ()
            return outputs
        else:
            if visual is None:
                raise ValueError('input frames are None when modality contains visual')
            if audio is None:
                raise ValueError('input audio are None when modality contains audio')

            ## question features
            qst_feature = self.question_encoder(question)
            xq = qst_feature.unsqueeze(0)

            ## audio features  B T,128
            audio_feat = F.relu(self.fc_a1(audio))
            audio_feat = self.fc_a2(audio_feat)

            visual_feat = visual.permute(1, 0, 2)
            audio_feat = audio_feat.permute(1, 0, 2)
            visual_feat_att = \
                self.attn_v(xq, visual_feat, visual_feat, attn_mask=None, key_padding_mask=None)[0].squeeze(0)
            src = self.linear12(self.dropout1(F.relu(self.linear11(visual_feat_att))))
            visual_feat_att = visual_feat_att + self.dropout2(src)
            visual_feat_att = self.norm1(visual_feat_att)
            audio_feat_att = self.attn_a(xq, audio_feat, audio_feat, attn_mask=None, key_padding_mask=None)[
                0].squeeze(0)
            src = self.linear22(self.dropout3(F.relu(self.linear21(audio_feat_att))))
            audio_feat_att = audio_feat_att + self.dropout4(src)
            audio_feat_att = self.norm2(audio_feat_att)

            feat = torch.cat((audio_feat_att, visual_feat_att), dim=-1)
            feat = self.tanh(feat)
            feat = self.fc_fusion(feat)

            ## fusion with question
            combined_feature = torch.mul(feat, qst_feature)
            combined_feature = self.tanh(combined_feature)
            out_qa = self.classifier(combined_feature)  # [batch_size, ans_vocab_size]
            outputs = ()

            if out_logits:
                outputs += (out_qa,)
            if out_features:
                if out_features_norm:
                    outputs += (F.normalize(combined_feature),)
                else:
                    outputs += (combined_feature,)
            if out_feature_before_fusion:
                outputs += (F.normalize(audio_feat_att), F.normalize(visual_feat_att),)
            if que_feature:
                outputs += (F.normalize(qst_feature),)
            if len(outputs) == 1:
                return outputs[0]
            else:
                return outputs


    def incremental_classifier(self, num_ans):
        weight = self.classifier.weight.data
        bias = self.classifier.bias.data
        in_features = self.classifier.in_features
        out_features = self.classifier.out_features
        self.classifier = nn.Linear(in_features, num_ans, bias=True)
        self.classifier.weight.data[:out_features] = weight
        self.classifier.bias.data[:out_features] = bias
        nn.init.kaiming_uniform_(self.classifier.weight[out_features:])
        self.classifier.bias.data[out_features:].zero_()


