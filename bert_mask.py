import torch
import json
from transformers import BertForMaskedLM
from transformers import AdamW, WarmupLinearSchedule
from transformers import BertTokenizer
from transformers import PYTORCH_PRETRAINED_BERT_CACHE
import argparse
import os
import random
import sys
import numpy as np
import torch.utils.data as data
from visdom import Visdom
from tqdm import tqdm, trange
import logging
import pdb


os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

np.set_printoptions(threshold=sys.maxsize)


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='semart_knowledge_filling_dataset/', type=str)
    parser.add_argument("--bert_model", default='bert-base-uncased', type=str)
    parser.add_argument("--do_lower_case", default=True)
    parser.add_argument('--seed', type=int, default=181)
    parser.add_argument("--learning_rate", default=8e-5, type=float)
    parser.add_argument("--num_train_epochs", default=10.0, type=float)
    parser.add_argument("--patience", default=3.0, type=float)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument("--device", default='cuda', type=str, help="cuda, cpu")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--eval_batch_size", default=64, type=int)
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--workers", default=8)
    parser.add_argument("--do_plot", default=True)
    parser.add_argument("--train_name", default="BERT-FillingBlank-SEP")
    args, unknown = parser.parse_known_args()
    return args


# For plotting graphs
class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom(port=8999)
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x, x]), Y=np.array([y, y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]),
                          env=self.env, win=self.plots[var_name], name=split_name, update='append')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Class to contain a single instance of the dataset
class DataSample(object):
    def __init__(self, template_sent, original_sent, candidate_words):
        self.template = template_sent
        self.original = original_sent
        self.candidates = candidate_words


# Dataloader class
class FillingDataset(data.Dataset):

    def __init__(self, args, split, tokenizer):
        # Params
        self.tokenizer = tokenizer
        self.max_seq_length = args.max_seq_length
        self.split = split

        # load data and preprocess
        data = self.data_load(args)
        self.samples = self.data_preprocess(data)
        self.num_samples = len(self.samples)
        logger.info('Loaded data with %d samples' % self.num_samples)

    def data_load(self, args):
        if self.split =='train':
            filename = os.path.join(args.data_dir, 'TRAIN_data.json')
        elif self.split == 'val':
            filename = os.path.join(args.data_dir, 'VAL_data.json')
        elif self.split == 'test':
            filename = os.path.join(args.data_dir, 'TEST_data.json')
        with open(filename, 'r') as f:
            data = json.load(f)
        return data

    def data_preprocess(self, data):
        samples = []
        for s in data:
            template = s[0]
            candidates = s[1] = s[1]
            original = s[2]
            # Remove <end> from template and candidates
            template = template.rsplit(' ', 1)[0]
            original = original.rsplit(' ', 1)[0]
            # Candidate words --> convert to string
            list_candidates = [word for type in candidates for word in candidates[type]]
            string_candidates = ', '.join(list_candidates)
            # Add to the list of samples
            samples.append(DataSample(template_sent=template, candidate_words=string_candidates, original_sent=original))
        return samples

    def __len__(self):
        return self.num_samples

    def truncate(self, template_tokens, candidates_tokens, max_len):
        while True:
            total_length = len(template_tokens) + len(candidates_tokens)
            if total_length <= max_len:
                break
            else:
                candidates_tokens.pop()

    def check_mask_start(self, input_ids, tokens, input_pos, mask_start):
        # the vocab id of '_' is 1035
        if input_ids[input_pos + 1] == 1035:
            # facing the first token of the mask, let the input_pos skip (+2 because of the underline '_')
            return 2
        elif input_ids[input_pos + 2] == 1035 and tokens[input_pos] == 'mis' and tokens[input_pos + 1] == '##c':
            # MISC mask will be tokenized into two tokens
            return 3
        elif input_ids[input_pos + 3] == 1035 and tokens[input_pos] == 'or' and tokens[input_pos + 1] == '##dina' and tokens[input_pos + 2] == '##l':
            # ORDINAL mask will be tokenized into two tokens
            return 4
        else:
            if mask_start is False:
                return -1
            else:
                return 0

    def __getitem__append_candidates_noa_original(self, index):
        # Input sequences: [CLS] + template + [SEP] + candidates + [SEP]
        #  Target sequence: [CLS] + original + [SEP]

        sample = self.samples[index]

        # Tokenize each string individually
        template_tokens = self.tokenizer.tokenize(sample.template)
        original_tokens = self.tokenizer.tokenize(sample.original)
        candidates_tokens = self.tokenizer.tokenize(sample.candidates)

        # Truncate "candidate_tokens" (inplace) if longer than max_seq_lenght (leave 3 tokens for [CLS], [SEP] and [SEP])
        self.truncate(template_tokens, candidates_tokens, self.max_seq_length - 3)

        # Construct input sequence of tokens
        tokens = [self.tokenizer.cls_token] + template_tokens + [self.tokenizer.sep_token] + candidates_tokens + [self.tokenizer.sep_token]
        segment_ids = [0] * (len(template_tokens) + 2) + [1] * (len(candidates_tokens) + 1)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding = [self.tokenizer.pad_token_id] * (self.max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length

        # Construct output sequence of tokens
        output_tokens = [self.tokenizer.cls_token] + original_tokens + [self.tokenizer.sep_token]
        output_ids = self.tokenizer.convert_tokens_to_ids(output_tokens)
        output_padding = [-1] * (self.max_seq_length - len(output_ids)) # -1 to ignore tokens that are not in output sequence
        output_ids += output_padding
        assert len(output_ids) == self.max_seq_length

        # Convert to tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        output_ids = torch.tensor(output_ids, dtype=torch.long)
        return input_ids, input_mask, segment_ids, output_ids

    def __getitem__sep_candidates(self, index):
        # Input sequences: [CLS] + template + [SEP] + candidates + [SEP]
        # Target sequence: [CLS] + original + [SEP]

        sample = self.samples[index]

        # Tokenize each string individually
        template_tokens = self.tokenizer.tokenize(sample.template)
        original_tokens = self.tokenizer.tokenize(sample.original)
        candidates_tokens = []
        split_candidates = str(sample.candidates).strip().split(',')
        for k, cand in enumerate(split_candidates):
            cand = cand.strip()
            cand_token = self.tokenizer.tokenize(cand) + [self.tokenizer.sep_token]
            candidates_tokens.extend(cand_token)

        self.truncate(template_tokens, candidates_tokens, self.max_seq_length - 3)

        # Construct input sequence of tokens
        tokens = [self.tokenizer.cls_token] + template_tokens + [self.tokenizer.sep_token] + candidates_tokens
        segment_ids = [0] * (len(template_tokens) + 2) + [1] * len(candidates_tokens)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding = [self.tokenizer.pad_token_id] * (self.max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length

        # Construct output sequence of tokens
        output_tokens = [self.tokenizer.cls_token] + original_tokens + [self.tokenizer.sep_token]
        output_ids = self.tokenizer.convert_tokens_to_ids(output_tokens)
        output_padding = [-1] * (self.max_seq_length - len(output_ids)) # -1 to ignore tokens that are not in output sequence
        output_ids += output_padding
        assert len(output_ids) == self.max_seq_length

        # Convert to tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        output_ids = torch.tensor(output_ids, dtype=torch.long)
        return input_ids, input_mask, segment_ids, output_ids

    def __getitem__(self, index):
        # Input sequences: [CLS] + template + [SEP] + candidates + [SEP]
        #  Target sequence: [CLS] + original + [SEP]

        sample = self.samples[index]

        # Tokenize each string individually
        template_tokens = self.tokenizer.tokenize(sample.template)
        original_tokens = self.tokenizer.tokenize(sample.original)
        candidates_tokens = self.tokenizer.tokenize(sample.candidates)

        # Truncate "candidate_tokens" (inplace) if longer than max_seq_lenght (leave 3 tokens for [CLS], [SEP] and [SEP])
        self.truncate(template_tokens, candidates_tokens, self.max_seq_length - 3)

        # Construct input sequence of tokens
        tokens = [self.tokenizer.cls_token] + template_tokens + [self.tokenizer.sep_token] + candidates_tokens + [self.tokenizer.sep_token]
        input_ids_length_before_pad = len(template_tokens) + 2
        segment_ids = [0] * (len(template_tokens) + 2) + [1] * (len(candidates_tokens) + 1)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding = [self.tokenizer.pad_token_id] * (self.max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length

        # Construct output sequence of tokens
        output_tokens = [self.tokenizer.cls_token] + original_tokens + [self.tokenizer.sep_token]
        output_ids = self.tokenizer.convert_tokens_to_ids(output_tokens)
        output_ids_masked = self.tokenizer.convert_tokens_to_ids(output_tokens)
        output_ids_length_before_pad = len(output_ids)

        mask_start = False            # record whether this position is the first token of a mask
        abnormal_flag = False           # track abnormal instances
        input_pos, output_pos = 0, 0    # two pointers for matching input and output of different lengths
        while True:
            if input_pos >= input_ids_length_before_pad or output_pos >= output_ids_length_before_pad:
                # the two pointers should reach the these ends at the same time
                assert input_pos == input_ids_length_before_pad and output_pos == output_ids_length_before_pad
                break
            elif input_ids[input_pos] == output_ids_masked[output_pos]:
                input_pos += 1
                output_pos += 1
                mask_start = False
            else:
                if mask_start is False:
                    while True:     # to deal with consecutive masks
                        offset = self.check_mask_start(input_ids, tokens, input_pos, mask_start)
                        if offset == -1:
                            abnormal_flag = True    # a very very rare case that lead to mis-matching, 0.15% in the dataset, so just skip them
                            break
                        if offset == 0:
                            break
                        input_pos += offset
                        mask_start = True
                output_ids_masked[output_pos] = -1
                output_pos += 1

        if abnormal_flag:
            output_padding = [-1] * (self.max_seq_length - len(output_ids))  # -1 to ignore tokens that are not in output sequence
            output_ids += output_padding
        else:
            output_padding = [-1] * (self.max_seq_length - len(output_ids_masked))  # -1 to ignore tokens that are not in output sequence
            output_ids = output_ids_masked + output_padding

        assert len(output_ids) == self.max_seq_length

        # Convert to tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        output_ids = torch.tensor(output_ids, dtype=torch.long)
        return input_ids, input_mask, segment_ids, output_ids


def train_epoch(args, model, train_dataloader, optimizer, max_grad_norm, scheduler, n_gpu, epoch):
    losses = AverageMeter()
    model.train()
    for step, batch in enumerate(tqdm(train_dataloader, desc="Train iter")):
        batch = tuple(t.to(args.device) for t in batch)
        input_ids, input_mask, segment_ids, output_ids = batch
        outputs = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, masked_lm_labels=output_ids)
        loss, prediction_scores = outputs[:2]
        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        losses.update(loss.item(), input_ids.shape[0])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    if args.do_plot:
        plotter.plot('loss', 'train', 'Loss', epoch, losses.avg)


def val_epoch(args, model, val_dataloader, n_gpu, epoch):
    losses = AverageMeter()
    model.eval()
    for step, batch in enumerate(tqdm(val_dataloader, desc="Val iter")):
        batch = tuple(t.to(args.device) for t in batch)
        input_ids, input_mask, segment_ids, output_ids = batch
        with torch.no_grad():
            outputs = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, masked_lm_labels=output_ids)
            loss, prediction_scores = outputs[:2]
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
        losses.update(loss.item(), input_ids.shape[0])
    if args.do_plot:
        plotter.plot('loss', 'val', 'Loss', epoch, losses.avg)
    return losses.avg


def train(args):

    # Create training directory
    modeldir = os.path.join('TrainingMASK', args.train_name)
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)

    # Prepare GPUs
    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(args.device, n_gpu))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # Load BERT pre-trained tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # Prepare model (in our case, BertForMaskedLM should do the job!)
    model = BertForMaskedLM.from_pretrained(args.bert_model, cache_dir=os.path.join(PYTORCH_PRETRAINED_BERT_CACHE._str,
                                                                                    'distributed_{}'.format(-1)))
    model.to(args.device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Load training data
    trainDataObject = FillingDataset(args, split='train', tokenizer=tokenizer)
    valDataObject = FillingDataset(args, split='val', tokenizer=tokenizer)
    train_dataloader = torch.utils.data.DataLoader(trainDataObject, batch_size=args.batch_size, shuffle=True,
                                                   pin_memory=True, num_workers=args.workers)
    val_dataloader = torch.utils.data.DataLoader(valDataObject, batch_size=args.eval_batch_size, shuffle=False,
                                                 pin_memory=True, num_workers=args.workers)
    num_train_optimization_steps = int(trainDataObject.num_samples / args.batch_size) * args.num_train_epochs

    # For visualization
    if args.do_plot:
        global plotter
        plotter = VisdomLinePlotter(env_name=args.train_name)

    # Optimizer
    num_warmup_steps = float(args.warmup_proportion) * float(num_train_optimization_steps)
    max_grad_norm = 1.0
    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=num_warmup_steps, t_total=num_train_optimization_steps)  # PyTorch scheduler

    # Start training
    logger.info('*** Start training ***')
    logger.info("Num examples = %d", train_dataloader.__len__())
    logger.info("Batch size = %d", args.batch_size)
    logger.info("Num steps = %d", num_train_optimization_steps)
    best_loss = float("Inf")
    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        train_epoch(args, model, train_dataloader, optimizer, max_grad_norm, scheduler, n_gpu, epoch)
        current_loss = val_epoch(args, model, val_dataloader, n_gpu, epoch)

        # Save a trained model and the associated configuration
        is_best = current_loss < best_loss
        if current_loss < best_loss:
            best_loss = current_loss
        if is_best:
            model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(modeldir)


def test_samples(args):
    modeldir = os.path.join('TrainingSEP', args.train_name)
    # Prepare GPUs
    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(args.device, n_gpu))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # Load BERT pre-trained tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # Prepare model (in our case, BertForMaskedLM should do the job!)
    model = BertForMaskedLM.from_pretrained(modeldir, cache_dir=os.path.join(PYTORCH_PRETRAINED_BERT_CACHE._str,
                                                                             'distributed_{}'.format(-1)))
    model.to(args.device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Load training data
    valDataObject = FillingDataset(args, split='val', tokenizer=tokenizer)
    val_dataloader = torch.utils.data.DataLoader(valDataObject, batch_size=args.eval_batch_size, shuffle=False,
                                                 pin_memory=True, num_workers=args.workers)

    input_sentences_and_candidates = []
    expected_outputs = []
    prediction_sentences = []
    model.eval()
    for step, batch in enumerate(tqdm(val_dataloader, desc="Val iter")):
        batch = tuple(t.to(args.device) for t in batch)
        input_ids, input_mask, segment_ids, output_ids = batch
        with torch.no_grad():
            outputs = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                            masked_lm_labels=output_ids)
            loss, prediction_scores = outputs[:2]
            prediction_ids = torch.argmax(prediction_scores, dim=2, keepdim=False)

            for idx in range(input_ids.size(0)):
                # input
                input_id_one = input_ids[idx].cpu().data.numpy().tolist()
                end_idx = input_id_one.index(0)
                input_id_one = input_id_one[:end_idx]
                input_tokens = tokenizer.convert_ids_to_tokens(input_id_one)
                input_sent = tokenizer.convert_tokens_to_string(input_tokens)
                input_sentences_and_candidates.append(input_sent)

                # expected output
                label_ids = output_ids[idx].cpu().data.numpy().tolist()
                end_idx = label_ids.index(-1)
                label_ids = label_ids[:end_idx]
                label_tokens = tokenizer.convert_ids_to_tokens(label_ids)
                label_sent = tokenizer.convert_tokens_to_string(label_tokens)
                expected_outputs.append(label_sent)

                # pred
                pred_ids = prediction_ids[idx].cpu().data.numpy().tolist()
                pred_tokens = tokenizer.convert_ids_to_tokens(pred_ids)
                pred_sent = tokenizer.convert_tokens_to_string(pred_tokens)
                prediction_sentences.append(pred_sent)

    print('Writing to file.')
    with open('filling_output_SEP/input.json', 'w') as j:
        json.dump(input_sentences_and_candidates, j)
    with open('filling_output_SEP/label.json', 'w') as j:
        json.dump(expected_outputs, j)
    with open('filling_output_SEP/output.json', 'w') as j:
        json.dump(prediction_sentences, j)
    print('Done!')


def show_samples(num_samples=10, seed=123):
    random.seed(seed)
    with open('filling_output_SEP/input.json', 'r') as j:
        input_sentences_and_candidates = json.load(j)
    with open('filling_output_SEP/label.json', 'r') as j:
        expected_outputs = json.load(j)
    with open('filling_output_SEP/output.json', 'r') as j:
        prediction_sentences = json.load(j)
    indices = random.sample(list(range(len(input_sentences_and_candidates))), k=num_samples)
    for i, idx in enumerate(indices):
        print('{}-th Result:\n Input template and candidate words: {}\n'
              'Expected output: {}\n'
              'Prediction: {}'.format(i, input_sentences_and_candidates[idx],
                                      expected_outputs[idx],
                                      prediction_sentences[idx]))
        print('*'*100)


if __name__ == "__main__":
    args = get_params()
    train(args)
    # test_samples(args)
    # show_samples()

