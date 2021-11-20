# Copyright (c) 2019, Facebook, Inc. and its affiliates. All Rights Reserved
"""

[ ] deal with chunks: need to useless chunks
[ ] implement models: 
[ ] impelment evaluation function
[ ] dataloading


"""

from __future__ import absolute_import, division, print_function

import wandb
import argparse
import collections
import json
import logging
import os
import random
import time
import gzip
import datetime
from io import open

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import get_scheduler

from transformers import BertTokenizer
from transformers import AdamW

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

PRED_FILE = "predictions.json"
EVAL_FILE = "eval_results.txt"
TEST_FILE = "test_results.txt"


class MRQAExample(object):
    """
    A single training/test example for the MRQA dataset.
    For examples without an answer, the start and end position are -1.
    """
    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.end_position:
            s += ", end_position: %d" % (self.end_position)
        return s


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position


# new function to deal with .gz and .jsonl file
def get_data(input_file):
    if input_file.endswith('.gz'):
        with gzip.GzipFile(input_file, 'r') as reader:
            # skip header
            content = reader.read().decode('utf-8').strip().split('\n')[1:]
            input_data = [json.loads(line) for line in content]
    else:
        with open(input_file, 'r', encoding="utf-8") as reader:
            # lines = reader.readlines()
            # input_data = [json.loads(line) for line in lines]
            print(reader.readline())
            input_data = [json.loads(line) for line in reader]
    return input_data


def read_mrqa_examples(input_file, is_training, ignore=0, percentage=1):
    """Read a MRQA json file into a list of MRQAExample."""
    input_data = get_data(input_file)

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    num_answers = 0
    num_to_ignore = int(ignore * len(input_data))
    num_to_load = int(percentage * len(input_data))
    if ignore != 0 and percentage != 1 and ignore + percentage == 1:
        num_to_load = max(num_to_load, len(input_data) - num_to_ignore)
    logger.info('Notes: # documents loaded = {}'.format(num_to_load - num_to_ignore))
    for entry in input_data[num_to_ignore:(num_to_ignore + num_to_load)]:
        paragraph_text = entry["context"]
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        for c in paragraph_text:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)
        for qa in entry["qas"]:
            qas_id = qa["qid"]
            question_text = qa["question"]
            start_position = None
            end_position = None
            orig_answer_text = None
            if is_training:
                answers = qa["detected_answers"]
                # import ipdb
                # ipdb.set_trace()
                spans = sorted([span for spans in answers for span in spans['char_spans']])
                # take first span
                char_start, char_end = spans[0][0], spans[0][1]
                orig_answer_text = paragraph_text[char_start:char_end + 1]
                start_position, end_position = char_to_word_offset[char_start], char_to_word_offset[
                    char_end]
                num_answers += sum([len(spans['char_spans']) for spans in answers])
            example = MRQAExample(qas_id=qas_id,
                                  question_text=question_text,
                                  doc_tokens=doc_tokens,
                                  orig_answer_text=orig_answer_text,
                                  start_position=start_position,
                                  end_position=end_position)
            examples.append(example)
    logger.info('Num avg answers: {}'.format(num_answers / len(examples)))
    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length, doc_stride, max_query_length,
                                 is_training):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000

    features = []
    for (example_index, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None
        if is_training:
            tok_start_position = -1
            tok_end_position = -1
        if is_training:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position,
             tok_end_position) = _improve_answer_span(all_doc_tokens, tok_start_position,
                                                      tok_end_position, tokenizer,
                                                      example.orig_answer_text)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3
        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index, split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)
            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            start_position = None
            end_position = None
            if is_training:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                else:
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset
            # NOTE changed: used to be < 5
            if example_index < 0:
                logger.info("*** Example ***")
                logger.info("unique_id: %s" % (unique_id))
                logger.info("example_index: %s" % (example_index))
                logger.info("doc_span_index: %s" % (doc_span_index))
                logger.info("tokens: %s" % " ".join(tokens))
                logger.info("token_to_orig_map: %s" %
                            " ".join(["%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
                logger.info("token_is_max_context: %s" %
                            " ".join(["%d:%s" % (x, y) for (x, y) in token_is_max_context.items()]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                if is_training:
                    answer_text = " ".join(tokens[start_position:(end_position + 1)])
                    logger.info("start_position: %d" % (start_position))
                    logger.info("end_position: %d" % (end_position))
                    logger.info("answer: %s" % (answer_text))

            features.append(
                InputFeatures(unique_id=unique_id,
                              example_index=example_index,
                              doc_span_index=doc_span_index,
                              tokens=tokens,
                              token_to_orig_map=token_to_orig_map,
                              token_is_max_context=token_is_max_context,
                              input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              start_position=start_position,
                              end_position=end_position))
            unique_id += 1

    return features


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits"])


def load_initialization(model, args):
    if os.path.exists(args.initialize_model_from_checkpoint + '/pytorch_model.bin'):
        ckpt = torch.load(args.initialize_model_from_checkpoint + '/pytorch_model.bin')
        model.load_state_dict(ckpt)
    else:
        ckpt = torch.load(args.initialize_model_from_checkpoint + '/saved_checkpoint')
        assert args.model == ckpt['args']['model'], args.model + ' vs ' + ckpt['args']['model']
        # if args.do_train and not args.transfer:
        #     assert args.percentage_train_data_to_ignore == ckpt['args']['percentage_train_data']
        model.load_state_dict(ckpt['model_state_dict'])
    logger.info("***** Model Initialization *****")
    logger.info("Loaded the model state from a saved checkpoint {}".format(
        args.initialize_model_from_checkpoint))


def turn_off_dropout(m):
    for mod in m.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = 0


def tune_bias_only(m):
    for name, param in m.bert.named_parameters():
        if 'bias' in name or 'LayerNorm' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def classify():
    raise NotImplementedError


def main(args):
    args.timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {}, n_gpu: {}".format(device, n_gpu))
    args.n_gpu = n_gpu

    # set up random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # deal with gradient accumulation
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    # actual bs = bs // g i.e. 5 = 10 // 2
    args.train_batch_size = \
        args.train_batch_size // args.gradient_accumulation_steps

    # argparse checkers
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")
    if args.do_train:
        assert args.train_file is not None
    if args.eval_test:
        assert args.test_file is not None
    # only evaluate on the test set: need an initialization
    if args.eval_test and not args.do_train:
        assert args.initialize_model_from_checkpoint is not None

    # set up logging files
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # set up the logging for this experiment
    args.output_dir += '/' + args.timestamp
    os.makedirs(args.output_dir)
    if args.do_train:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "train.log"), 'w'))
    else:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "eval.log"), 'w'))
    # log args
    logger.info(args)

    tokenizer = BertTokenizer.from_pretrained(args.model, do_lower_case=args.do_lower_case)

    if args.do_train and args.do_eval:
        # load dev dataset
        eval_dataset = get_data(input_file=args.dev_file)
        eval_examples = read_mrqa_examples(input_file=args.dev_file, is_training=False)
        eval_features = convert_examples_to_features(examples=eval_examples,
                                                     tokenizer=tokenizer,
                                                     max_seq_length=args.max_seq_length,
                                                     doc_stride=args.doc_stride,
                                                     max_query_length=args.max_query_length,
                                                     is_training=False)
        logger.info("***** Dev *****")
        logger.info("  Num orig examples = %d", len(eval_examples))
        logger.info("  Num split examples = %d", len(eval_features))
        logger.info("  Batch size = %d", args.eval_batch_size)
        args.dev_num_orig_ex = len(eval_examples)
        args.dev_num_split_ex = len(eval_features)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
        eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size)

    if args.do_train:
        train_examples = read_mrqa_examples(input_file=args.train_file,
                                            is_training=True,
                                            ignore=args.percentage_train_data_to_ignore,
                                            percentage=args.percentage_train_data)
        train_features = convert_examples_to_features(examples=train_examples,
                                                      tokenizer=tokenizer,
                                                      max_seq_length=args.max_seq_length,
                                                      doc_stride=args.doc_stride,
                                                      max_query_length=args.max_query_length,
                                                      is_training=True)

        if args.train_mode == 'sorted' or args.train_mode == 'random_sorted':
            train_features = sorted(train_features, key=lambda f: np.sum(f.input_mask))
        else:
            random.shuffle(train_features)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_start_positions = torch.tensor([f.start_position for f in train_features],
                                           dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                   all_start_positions, all_end_positions)
        train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size)
        train_batches = [batch for batch in train_dataloader]

        num_train_optimization_steps = \
            len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        logger.info("***** Train *****")
        logger.info("  Num orig examples = %d", len(train_examples))
        logger.info("  Num split examples = %d", len(train_features))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        args.train_num_orig_ex = len(train_examples)
        args.train_num_split_ex = len(train_features)

        eval_step = max(1, len(train_batches) // args.eval_per_epoch)
        best_result = None
        lrs = [args.learning_rate] if args.learning_rate else \
            [1e-4, 9e-5, 8e-5, 7e-5, 6e-5, 5e-5, 3e-5, 2e-5, 1e-5]
        # [5e-5, 3e-5, 2e-5, 1e-5, 5e-6, 3e-6, 2e-6, 1e-6]
        # [1e-6, 2e-6, 3e-6, 5e-6, 1e-5, 2e-5, 3e-5, 5e-5]
        for lr in lrs:
            # NOTE changed: removed the cache_dir, argument
            if args.initialize_model_from_checkpoint:
                model = load_model(model_type=args.model)
                load_initialization(model=model, args=args)
            else:
                model = load_model(model_type=args.model)

            if args.turn_off_dropout:
                turn_off_dropout(model)
            if args.tune_bias_only:
                tune_bias_only(model)

            classifier = None

            model.to(device)
            model.eval()
            classifier.to(device)

            if n_gpu > 1:
                model = torch.nn.DataParallel(model)
                classifier = torch.nn.DataParallel(classifier)
            param_optimizer = list(classifier.named_parameters())
            param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [{
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay':
                0.01
            }, {
                'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay':
                0.0
            }]
            optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
            lr_scheduler = get_scheduler(args.scheduler,
                                         optimizer=optimizer,
                                         num_warmup_steps=int(num_train_optimization_steps *
                                                              args.warmup_proportion),
                                         num_training_steps=num_train_optimization_steps)

            loss_fn = torch.nn.CrossEntropyLoss()

            if args.wandb:
                wandb.init(
                    project='spectral',
                    name=
                    f'{args.train_num_orig_ex}{args.dataset}_{args.model}_{args.scheduler}={lr}_{args.initialize_model_from_checkpoint}_{args.output_dir}',
                    notes=args.notes,
                    config=vars(args))
                wandb.watch(model)

            tr_loss = 0
            nb_tr_examples = 0
            nb_tr_steps = 0
            global_step = 0
            start_time = time.time()
            for epoch in range(int(args.num_train_epochs)):
                classifier.train()
                logger.info("Start epoch #{} (lr = {})...".format(epoch, lr))
                if args.train_mode == 'random' or args.train_mode == 'random_sorted':
                    random.shuffle(train_batches)
                for step, batch in enumerate(train_batches):
                    if n_gpu == 1:
                        batch = tuple(t.to(device) for t in batch)
                    else:
                        ???

                    # TODO: change this arugment
                    labels = None
                    embeds = model(batch=batch[:3])
                    probs = classifier(embeds)
                    loss = loss_fn(probs, labels)

                    if n_gpu > 1:
                        loss = loss.mean()
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    tr_loss += loss.item()
                    nb_tr_examples += labels.size(0)
                    nb_tr_steps += 1

                    loss.backward()
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                        global_step += 1

                    if args.wandb and (global_step + 1) % 25 == 0:
                        correct = torch.sum(torch.argmax(probs, dim=1).view(-1) == labels).item()
                        acc = 1.0 * correct / labels.size(0)
                        wandb.log({
                            '(Train) loss': loss.item(),
                            '(Train) acc': acc,
                        }, step=global_step)

                    if (step + 1) % eval_step == 0 or step + 1 == len(train_batches):
                        logger.info(
                            'Epoch: {}, Step: {} / {}, used_time = {:.2f}s, loss = {:.6f}'.format(
                                epoch, step + 1, len(train_batches),
                                time.time() - start_time, tr_loss / nb_tr_steps))
                        
                        if args.wandb:
                            wandb.log({
                                '(Train) loss': loss.item(),
                                ????
                            },
                                      step=global_step)

                        save_model = False
                        if args.do_eval:
                            # result, _, _ = \
                            #     evaluate(args, model, device, eval_dataset,
                            #              eval_dataloader, eval_examples, eval_features)
                            # TODO implement this function
                            result, __ = classify()
                            classifier.train()
                            if args.wandb:
                                wandb.log(result, step=global_step)
                            result['global_step'] = global_step
                            result['epoch'] = epoch
                            result['learning_rate'] = lr
                            result[
                                'batch_size'] = args.train_batch_size * args.gradient_accumulation_steps
                            result['eval_step'] = eval_step
                            if (best_result is None) or (result[args.eval_metric] >
                                                         best_result[args.eval_metric]):
                                best_result = result
                                # save model when getting new best result
                                save_model = True
                                logger.info(
                                    "!!! Best dev %s (lr=%s, epoch=%d): %.2f" %
                                    (args.eval_metric, str(lr), epoch, result[args.eval_metric]))
                            elif best_result is not None:
                                save_model = True
                        else:
                            # case: no evaluation so just save the latest model
                            save_model = True
                        if save_model:
                            # NOTE changed
                            # save the config
                            model.bert.config.to_json_file(
                                os.path.join(args.output_dir, 'config.json'))
                            # save the model
                            torch.save(
                                {
                                    'global_step': global_step,
                                    'args': vars(args),
                                    'classifier_state_dict': classifier.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                }, os.path.join(args.output_dir, 'saved_checkpoint'))
                            if best_result:
                                # i.e. best_result is not None
                                filename = EVAL_FILE
                                if len(lrs) != 1:
                                    filename = str(lr) + '_' + EVAL_FILE
                                with open(os.path.join(args.output_dir, filename), "w") as writer:
                                    for key in sorted(best_result.keys()):
                                        writer.write("%s = %s\n" % (key, str(best_result[key])))
                        # save checkpoint 
                        if args.save_checkpoint:
                            checkpoint = {
                                'global_step': global_step,
                                'args': vars(args),
                                'classifier_state_dict': classifier.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                            }
                            folder = args.output_dir + '/ckpt'
                            # create a folder if not existed
                            if not os.path.exists(folder):
                                os.makedirs(folder)
                            filename = folder + f'/{args.timestamp}_gstep={global_step}'
                            torch.save(checkpoint, filename)

    if args.eval_test:
        if args.wandb:
            wandb.init(
                project='spectral',
                name=
                f'{args.model}_{args.test_file}_{args.initialize_model_from_checkpoint}_{args.output_dir}',
                tags=['eval'],
                notes=args.notes,
                config=vars(args))

        eval_dataset = get_data(args.test_file)
        eval_examples = read_mrqa_examples(input_file=args.test_file, is_training=False)
        eval_features = convert_examples_to_features(examples=eval_examples,
                                                     tokenizer=tokenizer,
                                                     max_seq_length=args.max_seq_length,
                                                     doc_stride=args.doc_stride,
                                                     max_query_length=args.max_query_length,
                                                     is_training=False)
        logger.info("***** Test *****")
        logger.info("  Num orig examples = %d", len(eval_examples))
        logger.info("  Num split examples = %d", len(eval_features))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
        eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size)

        # NOTE change: only evaluate on the test set
        if not args.do_train:
            model = load_model(model_type=args.model)
            assert args.initialize_model_from_checkpoint is not None
            load_initialization(model=model, args=args)
            model.to(device)
            # TODO load classifier
            classifier = None
            classifier.to(device)
        # result, preds, nbest_preds = evaluate(args, model, device, eval_dataset, eval_dataloader,
        #                                       eval_examples, eval_features)
        results, preds = classify()
        with open(os.path.join(args.output_dir, PRED_FILE), "w") as writer:
            writer.write(json.dumps(preds, indent=4) + "\n")
        with open(os.path.join(args.output_dir, TEST_FILE), "w") as writer:
            for key in sorted(result.keys()):
                writer.write("%s = %s\n" % (key, str(result[key])))

        if args.wandb:
            wandb.log(result, step=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, type=str, required=True)
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--train_file", default=None, type=str)
    parser.add_argument("--dev_file", default=None, type=str)
    parser.add_argument("--test_file", default=None, type=str)
    parser.add_argument("--eval_per_epoch",
                        default=10,
                        type=int,
                        help="How many times it evaluates on dev set per epoch")
    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride",
                        default=128,
                        type=int,
                        help="When splitting up a long document into chunks, "
                        "how much stride to take between chunks.")
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum number of tokens for the question. Questions longer than this will "
        "be truncated to this length.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--eval_test",
                        action='store_true',
                        help='Wehther to run eval on the test set.')
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for predictions.")
    parser.add_argument("--learning_rate",
                        default=None,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--eval_metric", default='f1', type=str)
    parser.add_argument("--train_mode",
                        type=str,
                        default='random_sorted',
                        choices=['random', 'sorted', 'random_sorted'])
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
        "of training.")
    parser.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help="The total number of n-best predictions to generate in the nbest_predictions.json "
        "output file.")
    parser.add_argument("--max_answer_length",
                        default=30,
                        type=int,
                        help="The maximum length of an answer that can be generated. "
                        "This is needed because the start "
                        "and end predictions are not conditioned on one another.")
    parser.add_argument(
        "--verbose_logging",
        action='store_true',
        help="If true, all of the warnings related to data processing will be printed. "
        "A number of warnings are expected for a normal MRQA evaluation.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.")

    # below are customized arguments
    parser.add_argument('--wandb', action='store_true', help='Whether to use wandb for logging.')
    parser.add_argument('--notes', default='', help='Notes for this experiment: wandb logging')
    parser.add_argument(
        '--save_checkpoint',
        action='store_true',
        help=
        'Whether to save different checkpoints during training: recommend not to use this argument for space saving'
    )
    parser.add_argument('--percentage_train_data',
                        type=float,
                        default=1,
                        help='Percetage of training data to load: for debugging purpose')
    parser.add_argument('--initialize_model_from_checkpoint',
                        default=None,
                        help='Relative filepath to a saved checkpoint as model initialization.')
    parser.add_argument('--scheduler', default='linear', type=str, help='Learning rate scheduler.')
    parser.add_argument('--turn_off_dropout',
                        action='store_true',
                        help='Should rurn off dropout for simulation experiments')
    parser.add_argument("--dataset",
                        default=None,
                        type=str,
                        choices=['squad', 'hotpot', 'nq', 'trivia', 'search', 'news'])
    args = parser.parse_args()
    main(args)
