import torch
import torch.utils.data
from torch.utils.data import Dataset
import json
import re
from os.path import join
from PIL import Image
from utils import data_utils


class CLEVRXTrainDataset(Dataset):

    def __init__(self, path, img_dir, transform, tokenizer, max_seq_len):

        self.tokenizer = tokenizer
        self.transform = transform
        # question + <bos> The answer is <answer> becase <explanation> <eos>
        self.max_seq_len = max_seq_len
        self.data = json.load(open(path, 'r'))
        self.img_dir = img_dir
        self.ids_list = list(self.data.keys())

        for k, v in self.data.items():
            # some questions have more than one explanation
            # duplicate them for loading.
            # -1 because one explanation is already in ids_list
            if len(v['explanation']) > 1:
                self.ids_list += [str(k)] * (len(v['explanation']) - 1)

        self.index_tracker = {
            k: len(v['explanation']) - 1 for k, v in self.data.items()}

    def __getitem__(self, i):

        question_id = self.ids_list[i]
        sample = self.data[question_id]

        img_name = sample['image_name']
        split = re.search(r'CLEVR_(\w+)_\d+\.png', img_name).group(1)

        # question
        text_a = data_utils.proc_ques(sample['question'])
        # answer
        answer = sample['answer']

        # the index of the explanation for questions with multiple explanations
        exp_idx = self.index_tracker[question_id]
        if exp_idx > 0:
            self.index_tracker[question_id] -= 1    # decrease usage

        # explanation
        explanation = sample['explanation'][exp_idx]

        # tokenization process
        q_seg_id, a_seg_id, e_seg_id = self.tokenizer.convert_tokens_to_ids(
            ['<question>', '<answer>', '<explanation>'])

        q_tokens = self.tokenizer.tokenize(text_a)
        a_tokens = [self.tokenizer.bos_token] + \
            self.tokenizer.tokenize(" the answer is " + answer)
        e_tokens = self.tokenizer.tokenize(
            " because " + explanation) + [self.tokenizer.eos_token]
        tokens = q_tokens + a_tokens + e_tokens

        labels = q_tokens + a_tokens + e_tokens
        # we dont want to predict the question, set to pad to ignore in XE
        # labels will be shifted in the model, so for now set them same as tokens
        labels[:(len(q_tokens) + 1)] = [-100] * (len(q_tokens) + 1)

        q_ids = [q_seg_id] * len(q_tokens)
        a_ids = [a_seg_id] * len(a_tokens)
        e_ids = [e_seg_id] * len(e_tokens)
        segment_ids = q_ids + a_ids + e_ids

        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
            segment_ids = segment_ids[:self.max_seq_len]

        assert len(tokens) == len(segment_ids)
        assert len(tokens) == len(labels)

        # pad
        seq_len = len(tokens)
        padding_len = self.max_seq_len - seq_len
        tokens = tokens + ([self.tokenizer.pad_token] * padding_len)
        labels = labels + ([-100] * padding_len)
        segment_ids += ([e_seg_id] * padding_len)

        # convert tokens to ids
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        # convert tokens (!= -100) to ids
        labels = [self.tokenizer.convert_tokens_to_ids(
            t) if t != -100 else t for t in labels]
        labels = torch.tensor(labels, dtype=torch.long)

        segment_ids = torch.tensor(segment_ids, dtype=torch.long)

        # handle image
        img_path = join(self.img_dir, split, img_name)
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        qid = torch.LongTensor([int(question_id)])

        return (img, qid, input_ids, labels, segment_ids)

    def __len__(self):
        return len(self.ids_list)


class CLEVRXEvalDataset(Dataset):

    def __init__(self, path, img_dir, transform, tokenizer, max_seq_len):

        self.tokenizer = tokenizer
        self.transform = transform
        # question + <bos> The answer is <answer> becase <explanation> <eos>
        self.max_seq_len = max_seq_len
        self.data = json.load(open(path, 'r'))
        self.img_dir = img_dir
        self.ids_list = list(self.data.keys())

    def __getitem__(self, i):

        question_id = self.ids_list[i]
        sample = self.data[question_id]

        img_name = sample['image_name']
        split = re.search(r'CLEVR_(\w+)_\d+\.png', img_name).group(1)

        # question
        text_a = data_utils.proc_ques(sample['question'])

        # tokenization process
        q_seg_id, a_seg_id, e_seg_id = self.tokenizer.convert_tokens_to_ids(
            ['<question>', '<answer>', '<explanation>'])
        tokens = self.tokenizer.tokenize(text_a)
        segment_ids = [q_seg_id] * len(tokens)

        answer = [self.tokenizer.bos_token] + \
            self.tokenizer.tokenize(" the answer is")
        tokens += answer

        segment_ids += [a_seg_id] * len(answer)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)

        img_path = join(self.img_dir, split, img_name)
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        qid = torch.LongTensor([int(question_id)])

        return (img, qid, input_ids, segment_ids)

    def __len__(self):
        return len(self.ids_list)