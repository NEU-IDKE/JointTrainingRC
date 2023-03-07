import torch.nn as nn
import torch
from losses import N_Loss
from transformers import BertModel, RobertaModel
from torch.nn.utils.rnn import pad_sequence
from losses import TripletLoss, BinaryLoss, ClassfyLoss, InfoNce_Loss
import pickle
import torch.nn.functional as F


class RE_Model(nn.Module):
    def __init__(self, args, emd_size=768, block_size=64):
        super(RE_Model, self).__init__()
        self.args = args
        self.model = BertModel.from_pretrained(args.model_name_or_path)
        self.model.resize_token_embeddings(len(args.tokenizer))
        self.len_labels = args.num_class
        self.emd_size = emd_size
        self.block_size = block_size
        if self.args.context != 'no context':
            self.bilinear = nn.Linear(3*emd_size, len(args.label_list))
        else:
            self.bilinear = nn.Linear(2*emd_size, len(args.label_list))
        self.dropout_ = nn.Dropout(0.2)
        self.classifier = nn.Linear(2*emd_size, 1)



    def forward(self, input_ids=None, attention_mask=None, sh_pos=None, st_pos=None, th_pos=None, tt_pos=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        sequence_outputs = outputs[0]
        row_ix = torch.range(0, sequence_outputs.shape[0]-1).long()
        head = sequence_outputs[row_ix, sh_pos]
        tail = sequence_outputs[row_ix, th_pos]
        if self.args.context == 'local max':
            context = self.get_max_emb(sh_pos, st_pos, th_pos, tt_pos, sequence_outputs)
        elif self.args.context == 'global max':
            context = self.get_global_max(sequence_outputs, attention_mask)
        elif self.args.context == 'cls':
            context = sequence_outputs[:,0,:]
        
        if self.args.context != 'no context':
            b1 = torch.cat((head, tail, context), 1)
        else:
            b1 = torch.cat((head, tail), 1)
        b1 = self.dropout_(b1)
        logits = self.bilinear(b1)
        
        b2 = torch.cat((head,tail),1)
        b2 = self.dropout_(b2)
        logits1 = self.classifier(b2)
        logits1 = F.sigmoid(logits1)
        
        
        if labels is not None:
            loss_N = ClassfyLoss(self.args.loss_t)
            loss_B = BinaryLoss()
            
            loss = loss_N(logits, labels)
            loss1 = loss_B(logits1, labels)
            loss = loss + loss1
        else:
            loss = -1
#         return logits,loss
        return (logits, logits1), loss

    def get_global_max(self, sequence_outputs, attention_mask):
        max_embs,_ = torch.max(sequence_outputs,dim=1)
        return max_embs

    def get_max_emb(self, sh_pos, st_pos, th_pos, tt_pos, sequence_outputs):
        max_embs = []
        for i in range(sh_pos.shape[0]):
            if st_pos[i]+1 < th_pos[i]:
                start_index = st_pos[i]+1
                end_index = th_pos[i]
                max_emb, _ = torch.max(sequence_outputs[i][start_index:end_index], 0)
            elif tt_pos[i]+1 < sh_pos[i]:
                start_index = tt_pos[i]+1
                end_index = sh_pos[i]
                max_emb, _ = torch.max(sequence_outputs[i][start_index:end_index], 0)
            else:
                max_emb = torch.zeros(self.emd_size).cuda()
            max_embs.append(max_emb)
        max_embs = torch.stack(max_embs, 0)
        return max_embs.to(self.args.device)
