# Copyright (c) 2019, Facebook, Inc. and its affiliates. All Rights Reserved
"""

"""

from __future__ import absolute_import, division, print_function

import wandb
import argparse
import json
import logging
import os
import random
import time
import datetime
from io import open

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertModel, BertModel
from transformers import get_scheduler

from transformers import BertTokenizer
from transformers import AdamW

from model import SpectralClassifier, MLPClassifier, DCTClassifier

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

PRED_FILE = "predictions.json"
EVAL_FILE = "eval_results.txt"
TEST_FILE = "test_results.txt"


class Example(object):
    """
    A single training/test example for the MRQA dataset.
    For examples without an answer, the start and end position are -1.
    """
    def __init__(self, ex_id, input_text: str, label: int):
        self.ex_id = ex_id
        self.input_text = input_text
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "ex_id: %s" % (self.ex_id)
        s += ", input_text: %s" % (self.input_text)
        s += ", label: [%s]" % (" ".join(self.label))
        return s


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, unique_id, example_index, input_ids, input_mask, label):
        self.unique_id = unique_id
        self.example_index = example_index
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.label = label


# .jsonl file
def get_data(input_file):
    with open(input_file, 'r', encoding="utf-8") as reader:
        # lines = reader.readlines()
        # input_data = [json.loads(line) for line in lines]
        print(reader.readline())
        input_data = [json.loads(line) for line in reader]
    return input_data


def read_examples(input_file, is_training, ignore=0, percentage=1):
    """Read a json file into a list of Example."""
    input_data = get_data(input_file)

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    num_to_ignore = int(ignore * len(input_data))
    num_to_load = int(percentage * len(input_data))
    if ignore != 0 and percentage != 1 and ignore + percentage == 1:
        num_to_load = max(num_to_load, len(input_data) - num_to_ignore)
    logger.info('Notes: # examples loaded = {}'.format(num_to_load - num_to_ignore))
    for entry in input_data[num_to_ignore:(num_to_ignore + num_to_load)]:
        example = Example(ex_id=entry['ex_id'], input_text=entry['text'], label=entry['label'])
        examples.append(example)
    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000

    features = []
    for (example_index, example) in enumerate(examples):
        input_tokens = tokenizer.tokenize(example.input_text)

        if len(input_tokens) > max_seq_length:
            input_tokens = input_tokens[0:max_seq_length]

        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            # segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        # assert len(segment_ids) == max_seq_length

        features.append(
            InputFeatures(
                unique_id=unique_id,
                example_index=example.ex_id,
                input_ids=input_ids,
                input_mask=input_mask,
                # segment_ids=segment_ids,
                label=example.label))
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


def classify(args, model, classifier, device, eval_dataloader):
    classifier.eval()
    matrix = torch.zeros(2, 2)
    all_predictions = []
    for input_ids, input_mask, labels, example_indices in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=input_mask)
            probs = classifier(output, input_mask)
        preds = torch.argmax(probs, dim=1).view(-1)
        all_predictions.extend(preds.tolist())
        for i in range(preds.shape[0]):
            matrix[int(preds[i]), int(labels[i])] += 1
    total = torch.sum(matrix.view(-1)).item() * 1.0
    precision = (matrix[1, 1] / torch.sum(matrix[1, :])).item()
    recall = (matrix[1, 1] / torch.sum(matrix[:, 1])).item()
    results = {
        'accuracy': (matrix[0, 0] + matrix[1, 1]).item() / total,
        'precision': precision,
        'recall': recall,
        'f1': 2 * precision * recall / (precision + recall),
    }
    return results, all_predictions


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

    if args.attack == 'textfooler':
        if args.dataset == 'agnews':
            args.train_file = 'data/textfooler-bert-base-uncased-ag-news/train.jsonl'
            args.dev_file = 'data/textfooler-bert-base-uncased-ag-news/val.jsonl'
            args.test_file = 'data/textfooler-bert-base-uncased-ag-news/test.jsonl'

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
        eval_examples = read_examples(input_file=args.dev_file, is_training=False)
        eval_features = convert_examples_to_features(examples=eval_examples,
                                                     tokenizer=tokenizer,
                                                     max_seq_length=args.max_seq_length)
        logger.info("***** Dev *****")
        logger.info("  Num orig examples = %d", len(eval_examples))
        logger.info("  Num split examples = %d", len(eval_features))
        logger.info("  Batch size = %d", args.eval_batch_size)
        args.dev_num_orig_ex = len(eval_examples)
        args.dev_num_split_ex = len(eval_features)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in eval_features], dtype=torch.long)
        all_ex_ids = torch.tensor([f.example_index for f in eval_features], dtype=torch.long)
        # all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        # all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_labels, all_ex_ids)
        eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size)

    if args.do_train:
        train_examples = read_examples(input_file=args.train_file,
                                       is_training=True,
                                       percentage=args.percentage_train_data)
        train_features = convert_examples_to_features(examples=train_examples,
                                                      tokenizer=tokenizer,
                                                      max_seq_length=args.max_seq_length)

        if args.train_mode == 'sorted' or args.train_mode == 'random_sorted':
            train_features = sorted(train_features, key=lambda f: np.sum(f.input_mask))
        else:
            random.shuffle(train_features)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        # all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_labels)
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

            model = BertModel.from_pretrained(args.model)

            if args.classifier == 'mlp':
                classifier = MLPClassifier(embed_dim=model.config.hidden_size)
            elif args.classifier == 'spectral':
                classifier = SpectralClassifier(embed_dim=model.config.hidden_size,
                                                filter=args.filter)
            elif args.classifier == 'dct':
                classifier = DCTClassifier(embed_dim=model.config.hidden_size,
                                           max_seq_len=args.max_seq_length)

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
                    f'{args.train_num_orig_ex}{args.dataset}_{args.attack}_{args.model}_{args.classifier}_lr={lr}_seed={args.seed}_{args.output_dir}',
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
                        # TODO check if this is gonna run
                        batch = tuple(t.to(device) for t in batch)

                    input_ids, attention_mask, labels = batch
                    output = model(input_ids=input_ids, attention_mask=attention_mask)
                    probs = classifier(output, attention_mask)
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
                            '(Train) batch acc': acc,
                        },
                                  step=global_step)

                    if (step + 1) % eval_step == 0 or step + 1 == len(train_batches):
                        logger.info(
                            'Epoch: {}, Step: {} / {}, used_time = {:.2f}s, loss = {:.6f}'.format(
                                epoch, step + 1, len(train_batches),
                                time.time() - start_time, tr_loss / nb_tr_steps))

                        if args.wandb:
                            wandb.log({'(Train) loss': loss.item()}, step=global_step)

                        save_model = False
                        if args.do_eval:
                            result, __ = classify(args, model, classifier, device, eval_dataloader)
                            classifier.train()
                            if args.wandb:
                                wandb.log({'(Dev) ' + k: v
                                           for k, v in result.items()},
                                          step=global_step)
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
                            # # save the config
                            # model.config.to_json_file(os.path.join(args.output_dir, 'config.json'))
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
                        # save every check checkpoint
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
        # eval_dataset = get_data(args.test_file)
        eval_examples = read_examples(input_file=args.test_file, is_training=False)
        eval_features = convert_examples_to_features(examples=eval_examples,
                                                     tokenizer=tokenizer,
                                                     max_seq_length=args.max_seq_length)
        logger.info("***** Test *****")
        logger.info("  Num orig examples = %d", len(eval_examples))
        logger.info("  Num split examples = %d", len(eval_features))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in eval_features], dtype=torch.long)
        all_ex_ids = torch.tensor([f.example_index for f in eval_features], dtype=torch.long)
        # all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        # all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_labels, all_ex_ids)
        eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size)

        # # NOTE change: only evaluate on the test set
        # if not args.do_train:
        #     model = BertModel.from_pretrained(args.model)
        #     if args.classifier == 'mlp':
        #         classifier = MLPClassifier(embed_dim=model.config.hidden_size)
        #     elif args.classifier == 'spectral':
        #         classifier = SpectralClassifier(embed_dim=model.config.hidden_size,
        #                                         filter=args.filter)
        #     model.to(device)
        #     classifier.to(device)

        # load the best classifier saved
        ckpt = torch.load(os.path.join(args.output_dir, 'saved_checkpoint'))
        classifier.load_state_dict(ckpt['classifier_state_dict'])

        # result, preds, nbest_preds = evaluate(args, model, device, eval_dataset, eval_dataloader,
        #                                       eval_examples, eval_features)
        result, all_preds = classify(args, model, classifier, device, eval_dataloader)
        predictions = [x for x in zip(all_ex_ids.tolist(), all_preds)]
        with open(os.path.join(args.output_dir, PRED_FILE), "w") as writer:
            writer.write(json.dumps(predictions, indent=4) + "\n")
        with open(os.path.join(args.output_dir, TEST_FILE), "w") as writer:
            for key in sorted(result.keys()):
                writer.write("%s = %s\n" % (key, str(result[key])))

        if args.wandb:
            wandb.log({'(Test) ' + k: v for k, v in result.items()}, step=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, type=str, required=True)
    parser.add_argument("--classifier",
                        default=None,
                        type=str,
                        choices=['mlp', 'spectral', 'dct'],
                        required=True)
    parser.add_argument("--filter", default=None, type=str, choices=['low', 'mid', 'high'])
    parser.add_argument("--attack", default=None, type=str, choices=['textfooler', 'a2t', 'clara'])
    parser.add_argument("--dataset", default=None, type=str, choices=['snli', 'agnews', 'imdb'])
    parser.add_argument(
        "--output_dir",
        default='.experiment',
        type=str,
        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--train_file", default=None, type=str)
    parser.add_argument("--dev_file", default=None, type=str)
    parser.add_argument("--test_file", default=None, type=str)
    parser.add_argument("--eval_per_epoch",
                        default=3,
                        type=int,
                        help="How many times it evaluates on dev set per epoch")
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.")
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
                        default=40,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=10,
                        type=int,
                        help="Total batch size for predictions.")
    parser.add_argument("--learning_rate",
                        default=1e-3,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--eval_metric",
                        default='f1',
                        choices=['f1', 'accuracy', 'precision', 'recall'],
                        type=str)
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
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed', type=int, default=46, help="random seed for initialization")
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=4,
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
    parser.add_argument('--scheduler', default='linear', type=str, help='Learning rate scheduler.')
    args = parser.parse_args()
    main(args)
