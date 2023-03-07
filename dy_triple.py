import math
import torch
import numpy as np
from collections import Counter
from torch.utils import data
import json
import os
import random


def collate_fn1(batch):
    input_ids, attention_mask, label, sh_pos, st_pos, th_pos, tt_pos, max_length = map(torch.stack, zip(*batch))
    input_ids = input_ids.view(input_ids.shape[0]*2, -1)
    attention_mask = attention_mask.view(attention_mask.shape[0]*2, -1)
    max_len = torch.max(max_length).long().item()
    label = label.view(-1)
    sh_pos = sh_pos.view(-1)
    st_pos = st_pos.view(-1)
    th_pos = th_pos.view(-1)
    tt_pos = tt_pos.view(-1)
    # max_len = 256
    sh_pos[sh_pos >= max_len - 1] = max_len - 1
    st_pos[st_pos >= max_len - 1] = max_len - 1
    th_pos[th_pos >= max_len - 1] = max_len - 1
    tt_pos[tt_pos >= max_len - 1] = max_len - 1
    return dict(input_ids=input_ids[:, :max_len], attention_mask=attention_mask[:, :max_len], sh_pos=sh_pos,
                st_pos=st_pos, th_pos=th_pos, tt_pos=tt_pos, label=label)

class CPDataset(data.Dataset):
    """Overwritten class Dataset for model CP.

    This class prepare data for training of CP.
    """

    def __init__(self, args, features):
        """Inits tokenized sentence and positive pair for CP.

        """

        # print("The number of sentence in which tokenizer can't find head/tail entity is %d" % entityMarker.err)
        # Samples positive pair dynamically.
        self.input_ids = features['input_ids']
        self.attention_mask = features['attention_mask']
        self.entity_ids = features['entity_ids']
        self.label = features['label']
        self.max_length = features['max_length']
        self.args = args
        self.path = self.args.data_dir
        self.__sample__()

    def __pos_pair__(self, scope):
        """Generate positive pair.

        Args:
            scope: A scope in which all sentences' label are the same.
                scope example: [0, 12]

        Returns:
            all_pos_pair: All positive pairs.
            ! IMPORTTANT !
            Given that any sentence pair in scope is positive pair, there
            will be totoally (N-1)N/2 pairs, where N equals scope[1] - scope[0].
            The positive pair's number is proportional to N^2, which will cause
            instance imbalance. And If we consider all pair, there will be a huge
            number of positive pairs.
            So we sample positive pair which is proportional to N. And in different epoch,
            we resample sentence pair, i.e. dynamic sampling.
        """
        pos_scope = list(range(scope[0], scope[1]))

        # shuffle bag to get different pairs
        random.shuffle(pos_scope)
        all_pos_pair = []
        bag = []
        for i, index in enumerate(pos_scope):
            bag.append(index)
            if (i + 1) % 2 == 0:
                all_pos_pair.append(bag)
                bag = []
        return all_pos_pair

    def __sample__(self):
        """Samples positive pairs.

        After sampling, `self.pos_pair` is all pairs sampled.
        `self.pos_pair` example:
                [
                    [0,2],
                    [1,6],
                    [12,25],
                    ...
                ]
        """
        rel2scope = json.load(open(os.path.join(self.path, "rel2scope.json")))
        self.pos_pair = []
        for rel in rel2scope.keys():
            scope = rel2scope[rel]
            pos_pair = self.__pos_pair__(scope)
            self.pos_pair.extend(pos_pair)

        print("Postive pair's number is %d" % len(self.pos_pair))

    def __len__(self):
        """Number of instances in an epoch.

        Overwitten function.
        """
        return len(self.pos_pair)

    def __getitem__(self, index):
        """Get training instance.

        Overwitten function.

        Args:
            index: Instance index.

        Return:
            input: Tokenized word id.
            mask: Attention mask for bert. 0 means masking, 1 means not masking.
            label: Label for sentence.
            h_pos: Position of head entity.
            t_pos: Position of tail entity.
        """
        bag = self.pos_pair[index]
        inputs = torch.zeros(self.args.max_seq_length * 2)
        mask = torch.zeros(self.args.max_seq_length * 2)
        label = torch.zeros(2)
        sh_pos = torch.zeros(2)
        st_pos = torch.zeros(2)
        th_pos = torch.zeros(2)
        tt_pos = torch.zeros(2)
        max_len = torch.zeros(2)

        for i, ind in enumerate(bag):
            inputs[i * self.args.max_seq_length: (i + 1) * self.args.max_seq_length] = self.input_ids[ind]
            mask[i * self.args.max_seq_length: (i + 1) * self.args.max_seq_length] = self.attention_mask[ind]
            label[i] = self.label[ind]
            sh_pos[i] = self.entity_ids[ind][0]
            th_pos[i] = self.entity_ids[ind][2]
            st_pos[i] = self.entity_ids[ind][1]
            tt_pos[i] = self.entity_ids[ind][3]
            max_len[i] = self.max_length[ind]
        # sh_pos[sh_pos >= self.args.max_seq_length] = self.args.max_seq_length - 1
        # st_pos[st_pos >= self.args.max_seq_length] = self.args.max_seq_length - 1
        # th_pos[th_pos >= self.args.max_seq_length] = self.args.max_seq_length - 1
        # tt_pos[tt_pos >= self.args.max_seq_length] = self.args.max_seq_length - 1

        return inputs, mask, label, sh_pos, st_pos, th_pos, tt_pos, max_len