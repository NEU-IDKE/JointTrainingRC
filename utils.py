import os
import json
import tqdm
import torch
import random
import numpy as np

class InputExample(object):
    def __init__(self, id_, text, span_a, span_b, type_a, type_b, label):
        self.id = id_
        self.text = text
        self.span_a = span_a
        self.span_b = span_b
        self.type_a = type_a
        self.type_b = type_b
        self.label = label



class InputFeatures(object):
    def __init__(
        self,
        input_ids,
        attention_mask,
        entity_ids,
        label,
        max_seq_length
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.entity_ids = entity_ids
        self.label = label
        self.max_seq_length = max_seq_length


class DataProcessor(object):
    def get_train_data(self, args):
        return self._create_examples(args, 'train')

    def get_eval_data(self, args):
        return self._create_examples(args, 'dev')

    def get_test_data(self, args):
        return self._create_examples(args, 'test')

    def get_label_list(self, args):
        labels = set()
        for example in self.get_train_data(args):
            labels.add(example.label)
        labels.discard("no_relation")
        return ["no_relation"] + sorted(labels)

    def get_entiy_type(self, args):
        entity_type = set()
        path = os.path.join(args.data_dir, 'train.json')
        with open(path, 'r', encoding='utf8') as f:
            data = json.load(f)
        for i, item in enumerate(data):
            entity_type.add(item['subj_type'])
            entity_type.add(item['obj_type'])
        return entity_type

    def _create_examples(self, args, data_type):
        data_dir = os.path.join(args.data_dir, data_type + '.json')
        with open(data_dir, 'r', encoding='utf8') as f:
            data = json.load(f)
        examples = []
        for i, item in enumerate(data):
            tokens = item["token"]
            token_spans = dict(
                subj=(item["subj_start"], item["subj_end"] + 1), obj=(item["obj_start"], item["obj_end"] + 1)
            )

            if token_spans["subj"][0] < token_spans["obj"][0]:
                entity_order = ("subj", "obj")
            else:
                entity_order = ("obj", "subj")

            text = ""
            cur = 0
            char_spans = dict(subj=[None, None], obj=[None, None])
            for target_entity in entity_order:
                token_span = token_spans[target_entity]
                text += " ".join(tokens[cur: token_span[0]])
                if text:
                    text += " "
                char_spans[target_entity][0] = len(text)
                text += " ".join(tokens[token_span[0]: token_span[1]]) + " "
                char_spans[target_entity][1] = len(text)
                cur = token_span[1]
            text += " ".join(tokens[cur:])
            text = text.rstrip()

            examples.append(
                InputExample(
                    "%s-%s" % (data_type, i),
                    text,
                    char_spans["subj"],
                    char_spans["obj"],
                    item["subj_type"],
                    item["obj_type"],
                    item["relation"],
                )
            )
        return examples

def convert_examples_to_features(examples, tokenizer, max_length, label_list):
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for i, example in enumerate(examples):
        if example.span_a[1]<example.span_b[1]:
            span_order = ('span_a', 'span_b')
        else:
            span_order = ('span_b', 'span_a')
        cur = 0
        tokens = [tokenizer.cls_token]
        token_spans = {}
        for span_name in span_order:
            span = getattr(example, span_name)
            t = tokenizer.tokenize(example.text[cur: span[0]])
            tokens += t
            start = len(tokens)
            tokens.append('<H'+example.type_a+'>' if span_name == "span_a" else '<T'+example.type_b+'>')
            t1 = tokenizer.tokenize(example.text[span[0]: span[1]])
            tokens += t1
            tokens.append('<H/'+example.type_a+'>' if span_name == "span_a" else '<T/'+example.type_b+'>')
            token_spans[span_name] = (start, len(tokens))
            cur = span[1]

        t2 = tokenizer.tokenize(example.text[cur:])
        tokens += t2
        tokens = tokens[:max_length-1]
        attention_mask = [1] * len(tokens)
        tokens += [tokenizer.sep_token]
        max_seq_length = len(tokens)
        attention_mask += [1]
        if len(tokens) < max_length:
            pad_len = max_length-len(tokens)
            tokens += [tokenizer.pad_token]*pad_len
            # print(len(attention_mask))
            attention_mask += [0]*pad_len
            # print(len([0]*(max_length-len(tokens))))
            # print(len(attention_mask))

        # print(max_seq_length)
        # break
        # print(len(attention_mask))
        # print(len(label_mask))
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # print(tokens)
        # print(input_ids)
        # print(attention_mask)

        entity_index = []
        for a,b in sorted(token_spans.items(),key=lambda x:x[0]):
            b = list(b)
            b[-1] -= 1
            entity_index.extend(b)
        assert len(input_ids)==len(attention_mask)
        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                entity_ids=entity_index,
                label=label_map[example.label],
                max_seq_length = max_seq_length
            )
        )
    return features

def seed_everything(seed=42):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
def collate_fn(batch):
    input_ids, attention_mask, entity_ids, label, max_length= map(torch.stack, zip(*batch))
    max_len = max(max_length).item()
    input_ids = input_ids[:, :max_len]
    attention_mask = attention_mask[:, :max_len]
    entity_ids[entity_ids >= max_len-1] = max_len - 1
    return dict(
        input_ids=input_ids.long(),
        attention_mask=attention_mask.long(),
        sh_pos=entity_ids[:, 0].long(),
        st_pos=entity_ids[:, 1].long(),
        th_pos=entity_ids[:, 2].long(),
        tt_pos=entity_ids[:, 3].long(),
        label=label.long()
        )








