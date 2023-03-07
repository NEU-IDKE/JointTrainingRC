import argparse
from utils import DataProcessor, convert_examples_to_features, seed_everything, collate_fn
from model import RE_Model
from dy_triple import CPDataset, collate_fn1
from transformers import BertTokenizer, BertConfig, BertModel, RobertaTokenizer
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import torch
import tqdm
import os
import pickle
from logger import MyLog
logger = MyLog('log_nocontext_tacred.txt').get_log()


def load_examples(args, fold='train'):
    processor = DataProcessor()
    if fold == 'train':
        examples = processor.get_train_data(args)
    elif fold == 'dev':
        examples = processor.get_eval_data(args)
    else:
        examples = processor.get_test_data(args)
    label_list = processor.get_label_list(args)
    features = convert_examples_to_features(examples, args.tokenizer, args.max_seq_length, label_list)
    input_ids = torch.tensor([o.input_ids for o in features],dtype=torch.long)
    attention_mask = torch.tensor([o.attention_mask for o in features], dtype=torch.long)
    entity_ids = torch.tensor([o.entity_ids for o in features], dtype=torch.long)
    label = torch.tensor([o.label for o in features], dtype=torch.long)
    max_length = torch.tensor([o.max_seq_length for o in features], dtype=torch.long)


    if fold in ['dev', 'test']:
        features = TensorDataset(input_ids, attention_mask, entity_ids, label, max_length)
        sampler = SequentialSampler(features)
        data_loader = DataLoader(features, sampler=sampler, batch_size=args.test_batch_size, collate_fn=collate_fn)
    else:
        features = dict(input_ids=input_ids, attention_mask=attention_mask,
                        entity_ids=entity_ids, label=label, max_length=max_length)
        dataset = CPDataset(args, features)
        data_loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn1)
    # print(data_loader)
    return data_loader

def main(ind):
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./datasets/tacred/", type=str)
    parser.add_argument("--transformer_type", default="bert", type=str)
    parser.add_argument("--model_name_or_path", default="../../../pretrained_model/bert-base-uncased/", type=str)
    parser.add_argument("--train_file", default="train.json", type=str)
    parser.add_argument("--dev_file", default="dev.json", type=str)
    parser.add_argument("--test_file", default="test.json", type=str)
    parser.add_argument("--save_path", default="outputs/", type=str)
    parser.add_argument("--tokenizer_name", default="../../../pretrained_model/bert-base-uncased/vocab.txt", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--context", default='no context', type=str,
                        help="context type")
    parser.add_argument("--do_train", default=True, type=bool,
                        help="Batch size for training.")
    parser.add_argument("--do_eval", default=True, type=bool,
                        help="Batch size for training.")
    parser.add_argument("--do_test", default=True, type=bool,
                        help="Batch size for training.")
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=32, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.15, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--weight_decay", dest="weight_decay", type=float,
                        default=1e-5, help="weight decay")
    parser.add_argument("--num_train_epochs", default=5, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--seed", type=int, default=66,
                        help="random seed for initialization")
    parser.add_argument("--num_class", type=int, default=42,
                        help="Number of relation types in dataset.")
    args = parser.parse_args()
    seed_everything(args.seed)
    args.loss_t = ind
    args.tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name)
    # args.tokenizer.add_special_tokens({'additional_special_tokens': ["<HEAD>","<TAIL>","<HEAD/>","<TAIL/>"]})
    args.label_list = DataProcessor().get_label_list(args)
    args.entity_type = DataProcessor().get_entiy_type(args)
    special_tokens = []
    for e_t in args.entity_type:
        special_tokens.extend(["<H"+e_t+">","<T"+e_t+">","<H/"+e_t+">","<T/"+e_t+">"])
    special_tokens.sort()
    args.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.do_train:
        train_dataloader = load_examples(args, fold='train')
        re_model = RE_Model(args)
        re_model.to(args.device)
        train(re_model, train_dataloader, args)
    if args.do_eval:
        pass
    if args.do_test:
        checkpoint = os.path.join(args.save_path, 'best_checkpoint/context.ckpt')
        model = RE_Model(args)
        model.load_state_dict(torch.load(checkpoint))
        results = evaluate(model, args, data_type='dev')
#         predict(model, args, data_type='test')
        logger.info('测试集f1为{},recall为{},precision为{}'.format(results['f1'],
                                                                    results['recall'], results['precision']))
        logger.info('---------------------------{}----------------------------'.format(args.loss_t))

def train(model, data_loader, args):
    num_steps = len(data_loader) // args.gradient_accumulation_steps * args.num_train_epochs
    # new_layer = ['bilinear']
    # optimizer_grouped_parameter = (
    #     {"params":[p for n,p in model.named_parameters() if any(n in nd for nd in new_layer)], "lr":1e-4},
    #     {"params": [p for n, p in model.named_parameters() if not any(n in nd for nd in new_layer)]}
    # )
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameter = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameter, lr=args.learning_rate, eps=args.adam_epsilon)
    schduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_steps*args.warmup_ratio, num_training_steps=num_steps)
    best_eval_score = {'f1':0, 'precision':0, 'recall':0}

    for epoch in range(args.num_train_epochs):
        global_steps = 0
        model.zero_grad()
        for i,batch in tqdm.tqdm(enumerate(data_loader)):
            model.train()
            inputs = {
                'input_ids': batch['input_ids'].long().to(args.device),
                'attention_mask': batch['attention_mask'].long().to(args.device),
                'sh_pos': batch['sh_pos'].long().to(args.device),
                'st_pos': batch['st_pos'].long().to(args.device),
                'th_pos': batch['th_pos'].long().to(args.device),
                'tt_pos': batch['tt_pos'].long().to(args.device),
                'labels': batch['label'].long().to(args.device)
            }

            # s = 1
            # s += model
            outputs = model(**inputs)
            loss = outputs[1] / args.gradient_accumulation_steps
            loss.backward()
            global_steps += 1
            if i % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                schduler.step()
                model.zero_grad()
            if global_steps==len(data_loader):
                results = evaluate(model, args, 'dev')
                if results['f1']>best_eval_score['f1']:
                    best_eval_score['f1'] = results['f1']
                    best_eval_score['precision'] = results['precision']
                    best_eval_score['recall'] = results['recall']
                    output_dir = os.path.join(args.save_path, 'best_checkpoint')
                    if not os.path.exists(output_dir):
                        os.mkdir(output_dir)
                    torch.save(model.state_dict(), output_dir+'/context.ckpt')

                    test_results = evaluate(model, args, 'test')
                    logger.info('目前为止test最好的f1为{},recall为{},precision为{}'.format(test_results['f1'],
                                                                       test_results['recall'],test_results['precision']))

                logger.info('第{}epoch的{}step的f1为{},recall为{},precision为{}'.format(epoch, global_steps, results['f1'],
                                                                            results['recall'],results['precision']))
                logger.info('目前为止最好的f1为{},recall为{},precision为{}'.format(best_eval_score['f1'],
                                                                            best_eval_score['recall'],best_eval_score['precision']))
                logger.info('---------------------------{}----------------------------'.format(args.loss_t))
            torch.cuda.empty_cache()

def evaluate(model, args, data_type):
    dev_loader = load_examples(args, fold=data_type)
    model.to(args.device)
    model.eval()
    predictions = []
    labels = []
    for i, batch in enumerate(dev_loader):
        inputs = {
            'input_ids': batch['input_ids'].long().to(args.device),
            'attention_mask': batch['attention_mask'].long().to(args.device),
            'sh_pos': batch['sh_pos'].long().to(args.device),
            'st_pos': batch['st_pos'].long().to(args.device),
            'th_pos': batch['th_pos'].long().to(args.device),
            'tt_pos': batch['tt_pos'].long().to(args.device)
        }
        with torch.no_grad():
            logits,logits1 = model(**inputs)[0]
            logits1 = logits1.squeeze(1)
        prediction = []
        for j in range(logits.shape[0]):
            if logits1[j]<0.5:
                prediction.append(0)
            else:
                index = torch.argmax(logits[j].detach().cpu()).item()
                prediction.append(index)
        predictions.extend(prediction)
        # predictions.extend(logits.detach().cpu().numpy().argmax(axis=1))
        labels.extend(batch['label'].cpu().tolist())
    count1 = 0
    count2 = 0
    for xx,yy in zip(predictions,labels):
        if xx==0 and yy==0:
            count1 += 1
        if yy!=0 and yy==xx:
            count2 += 1
    logger.info('count1:{}'.format(count1))
    logger.info('count2:{}'.format(count2))
            
    num_predicted_labels = 0
    num_gold_labels = 0
    num_correct_labels = 0

    for label, prediction in zip(labels, predictions):
        if prediction != 0:
            num_predicted_labels += 1
        if label != 0:
            num_gold_labels += 1
            if prediction == label:
                num_correct_labels += 1

    if num_predicted_labels > 0:
        precision = num_correct_labels / num_predicted_labels
    else:
        precision = 0.0
    recall = num_correct_labels / num_gold_labels
    if recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return dict(precision=precision, recall=recall, f1=f1)

if __name__=="__main__":
    for ind in [0.9]:
        main(ind)




