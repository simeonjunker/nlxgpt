import torch
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from transformers import GPT2Tokenizer, AutoConfig
from transformers import AdamW, get_linear_schedule_with_warmup
import json
import re
from os.path import join
from cococaption.pycocotools.coco import COCO
from cococaption.pycocoevalcap.eval import COCOEvalCap
from PIL import Image
from accelerate import Accelerator
from models.gpt import GPT2LMHeadModel
from models.clip_vit import ImageEncoder
from utils.eval_utils import top_filtering
from utils import data_utils
import argparse
from os import getcwd
from time import gmtime, strftime


def change_requires_grad(model, req_grad):
    for p in model.parameters():
        p.requires_grad = req_grad


def load_checkpoint(ckpt_path, epoch):

    model_name = 'nle_model_{}'.format(str(epoch))
    tokenizer_name = 'nle_gpt2_tokenizer_0'
    filename = 'ckpt_stats_' + str(epoch) + '.tar'

    tokenizer = GPT2Tokenizer.from_pretrained(
        ckpt_path + tokenizer_name)        # load tokenizer
    model = GPT2LMHeadModel.from_pretrained(
        ckpt_path + model_name).to(device)   # load model with config
    opt = torch.load(ckpt_path + filename)
    optimizer = get_optimizer(model, learning_rate)
    optimizer.load_state_dict(opt['optimizer_state_dict'])
    start_epoch = opt['epoch'] + 1
    scheduler_dic = opt['scheduler']
    del opt
    torch.cuda.empty_cache()

    return tokenizer, model, optimizer, scheduler_dic, start_epoch


def load_pretrained():

    model_path = 'pretrained_model/model1/pretrain_model'
    tokenizer_path = 'pretrained_model/model1/pretrain_tokenizer_0'
    tokenizer = GPT2Tokenizer.from_pretrained(
        tokenizer_path)        # load tokenizer
    model = GPT2LMHeadModel.from_pretrained(
        model_path).to(device)   # load model with config
    return tokenizer, model


def save_checkpoint(
        epoch, unwrapped_model, optimizer,
        tokenizer, scheduler, ckpt_path, **kwargs):

    model_name = 'nle_model_{}'.format(str(epoch))
    tokenizer_name = 'nle_gpt2_tokenizer_{}'.format(str(epoch))
    filename = 'ckpt_stats_' + str(epoch) + '.tar'

    if epoch == 0:
        tokenizer.save_pretrained(
            ckpt_path + tokenizer_name)   # save tokenizer

    unwrapped_model.save_pretrained(
        ckpt_path + model_name, save_function=accelerator.save)

    opt = {'epoch': epoch,
           'optimizer_state_dict': optimizer.state_dict(),
           'scheduler': scheduler.state_dict(),
           **kwargs}

    print(f'save model checkpoint to {ckpt_path + filename}')

    accelerator.save(opt, ckpt_path + filename)

# def get_scores(annFile, resFile, save_scores_path):

#     coco = COCO(annFile)
#     cocoRes = coco.loadRes(resFile)
#     cocoEval = COCOEvalCap(coco, cocoRes)
#     cocoEval.evaluate()
#     with open(save_scores_path, 'w') as w:
#         json.dump(cocoEval.eval, w)


def filter_and_get_scores(
        resFileExp, save_scores_pathExp,
        full_predictions, exp_predictions):

    all_file = json.load(open(nle_data_val_path, 'r'))

    gt_answers = {}
    for key, value in all_file.items():
        gt_answers[int(key)] = value['answer']  # NOTE changed to single image

    pred_answers = {}
    for item in full_predictions:
        pred_answers[item['image_id']] = item['caption'].split("because")[
            0].strip()

    correct_keys = []
    for key, value in pred_answers.items():
        gt_answer = gt_answers[key]
        # to measure accuracy for VQA, please change "==" to "in" (if value in gt_answer:)
        if value == gt_answer:
            correct_keys.append(key)

    exp_preds = [
        item for item in exp_predictions if item['image_id'] in correct_keys]

    with open(resFileExp, 'w') as w:
        json.dump(exp_preds, w)

    coco = COCO(annFileExp)
    cocoRes = coco.loadRes(resFileExp)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    with open(save_scores_pathExp, 'w') as w:
        json.dump(cocoEval.eval, w)


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


def sample_sequences(model, tokenizer, loader, limit=None):

    model.eval()
    results_exp = []
    results_full = []
    SPECIAL_TOKENS = ['<|endoftext|>', '<pad>',
                      '<question>', '<answer>', '<explanation>']
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    because_token = tokenizer.convert_tokens_to_ids('Ġbecause')
    max_len = 20

    for i, batch in enumerate(loader):

        if limit:
            if i == limit:
                break

        current_output = []
        batch = tuple(input_tensor.to(device) for input_tensor in batch)
        img, img_id, input_ids, segment_ids = batch
        img_embeddings = image_encoder(img)
        always_exp = False

        with torch.no_grad():

            for step in range(max_len + 1):

                if step == max_len:
                    break

                outputs = model(input_ids=input_ids,
                                past_key_values=None,
                                attention_mask=None,
                                token_type_ids=segment_ids,
                                position_ids=None,
                                encoder_hidden_states=img_embeddings,
                                encoder_attention_mask=None,
                                labels=None,
                                use_cache=False,
                                return_dict=True)

                lm_logits = outputs.logits
                logits = lm_logits[0, -1, :] / temperature
                logits = top_filtering(logits, top_k=top_k, top_p=top_p)
                probs = F.softmax(logits, dim=-1)
                prev = torch.topk(probs, 1)[
                    1] if no_sample else torch.multinomial(probs, 1)

                if prev.item() in special_tokens_ids:
                    break

                # take care of when to start the <explanation> token
                if not always_exp:

                    if prev.item() != because_token:
                        new_segment = special_tokens_ids[-2]   # answer segment
                    else:
                        # explanation segment
                        new_segment = special_tokens_ids[-1]
                        always_exp = True
                else:
                    # explanation segment
                    new_segment = special_tokens_ids[-1]

                new_segment = torch.LongTensor([new_segment]).to(device)
                current_output.append(prev.item())
                input_ids = torch.cat((input_ids, prev.unsqueeze(0)), dim=1)
                segment_ids = torch.cat(
                    (segment_ids, new_segment.unsqueeze(0)), dim=1)

        decoded_sequences = tokenizer.decode(
            current_output, skip_special_tokens=True).lstrip()
        results_full.append(
            {"image_id": img_id.item(), "caption": decoded_sequences})

        if 'because' in decoded_sequences:
            cut_decoded_sequences = decoded_sequences.split(
                'because')[-1].strip()
        else:
            cut_decoded_sequences = " ".join(decoded_sequences.split()[2:])

        results_exp.append(
            {"image_id": img_id.item(), "caption": cut_decoded_sequences})
        print("\rEvaluation: Finished {}/{}".format(i,
              len(loader)), end='          ')

    return results_full, results_exp


def get_optimizer(model, learning_rate):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    return optimizer


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--finetune_pretrained', default=True)
    parser.add_argument('--eval_batch_size', default=1, type=int)
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--ckpt_path', default='ckpts/CLEVR-X/')
    parser.add_argument('--caption_save_path', default='cococaption/results/')
    parser.add_argument('--annFileExp', default='cococaption/annotations/clevrX_dev_annot_exp.json')
    parser.add_argument('--annFileFull', default='cococaption/annotations/clevrX_dev_annot_full.json')
    parser.add_argument('--nle_data_train_path', default='nle_data/CLEVR-X/clevrX_train.json')
    parser.add_argument('--nle_data_test_path', default='nle_data/CLEVR-X/clevrX_test.json')
    parser.add_argument('--nle_data_val_path', default='nle_data/CLEVR-X/clevrX_dev.json')
    parser.add_argument('--img_dir', default='/home/public/corpora/CLEVR/CLEVR_v1.0/images')
    parser.add_argument('--max_seq_len', default=40, type=int)
    parser.add_argument('--load_from_epoch', default=None, type=int)
    parser.add_argument('--no_sample', default=True, type=bool)
    parser.add_argument('--top_k', default=0, type=int)
    parser.add_argument('--top_p', default=0.9, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_train_epochs', default=30, type=int)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--temperature', default=1, type=float)
    parser.add_argument('--limit', default=None, type=int)
    parser.add_argument('--print_step', default=10, type=int)

    args = parser.parse_args()

    print('Train settings:')
    print(args)
    print(f'working directory: {getcwd()}')
    print(f'CUDA available: {torch.cuda.is_available()}')

    accelerator = Accelerator()
    device = accelerator.device

    print(f'Training on device: {device}')

    finetune_pretrained = args.finetune_pretrained
    eval_batch_size = args.eval_batch_size
    img_size = args.img_size
    ckpt_path = args.ckpt_path
    caption_save_path = args.caption_save_path
    annFileExp = args.annFileExp
    annFileFull = args.annFileFull
    nle_data_train_path = args.nle_data_train_path
    nle_data_test_path = args.nle_data_test_path
    nle_data_val_path = args.nle_data_val_path
    img_dir = args.img_dir
    max_seq_len = args.max_seq_len
    load_from_epoch = args.load_from_epoch
    no_sample = args.no_sample
    top_k = args.top_k
    top_p = args.top_p
    batch_size = args.batch_size
    num_train_epochs = args.num_train_epochs
    weight_decay = args.weight_decay
    gradient_accumulation_steps = args.gradient_accumulation_steps
    start_epoch = args.start_epoch
    temperature = args.temperature
    limit = args.limit
    print_step = args.print_step
    
    learning_rate = 2e-5 if not finetune_pretrained else 1e-5

    image_encoder = ImageEncoder(device).to(device)
    change_requires_grad(image_encoder, False)

    if load_from_epoch is not None:
        tokenizer, model, optimizer, scheduler_dic, start_epoch = load_checkpoint(
            ckpt_path, load_from_epoch)

    else:

        if finetune_pretrained:
            tokenizer, model = load_pretrained()
            optimizer = get_optimizer(model, learning_rate)
        else:
            tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
            orig_num_tokens = len(tokenizer.encoder)

            num_new_tokens = tokenizer.add_special_tokens({
                'pad_token': '<pad>',
                'additional_special_tokens': [
                    '<question>', '<answer>', '<explanation>']
            })

            assert len(tokenizer) == orig_num_tokens + num_new_tokens
            config = AutoConfig.from_pretrained('distilgpt2')

            # Add configs
            setattr(config, 'img_size', None)
            setattr(config, 'max_seq_len', None)
            config.img_size = img_size
            config.max_seq_len = max_seq_len
            config.add_cross_attention = True

            model = GPT2LMHeadModel.from_pretrained(
                'distilgpt2', config=config)
            model.resize_token_embeddings(len(tokenizer))
            model = model.to(device)
            optimizer = get_optimizer(model, learning_rate)

    print("Model Setup Ready...")

    img_transform = transforms.Compose([transforms.Resize((img_size, img_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])]
                                       )

    train_dataset = CLEVRXTrainDataset(path=nle_data_train_path,
                                       img_dir=img_dir,
                                       transform=img_transform,
                                       tokenizer=tokenizer,
                                       max_seq_len=max_seq_len)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True)

    val_dataset = CLEVRXEvalDataset(path=nle_data_val_path,
                                    img_dir=img_dir,
                                    transform=img_transform,
                                    tokenizer=tokenizer,
                                    max_seq_len=max_seq_len)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=True)

    model, optimizer, train_loader = accelerator.prepare(
        model, optimizer, train_loader)

    t_total = (len(train_loader) // gradient_accumulation_steps) * \
        num_train_epochs
    warmup_steps = 0   # 0.10 * t_total
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

    if load_from_epoch is not None:
        scheduler.load_state_dict(scheduler_dic)

    for epoch in range(start_epoch, num_train_epochs):

        model.train()
        accum_loss = 0

        for step, batch in enumerate(train_loader):

            if limit:
                if step == limit:
                    break

            batch = tuple(input_tensor.to(device) for input_tensor in batch)
            img, _, input_ids, labels, segment_ids = batch

            img_embeddings = image_encoder(img)

            outputs = model(input_ids=input_ids,
                            past_key_values=None,
                            attention_mask=None,
                            token_type_ids=segment_ids,
                            position_ids=None,
                            encoder_hidden_states=img_embeddings,
                            encoder_attention_mask=None,
                            labels=labels,
                            use_cache=False,
                            return_dict=True)

            loss = outputs.loss
            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)
            accum_loss += loss.item()

            if step % gradient_accumulation_steps == 0 or step == len(train_loader) - 1:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                if step % print_step == 0:
                    current_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
                    accelerator.print(f"""
                        \r{current_time} --- Epoch {epoch} / {num_train_epochs}, Iter {step} / {len(train_loader)}, Loss: {round(accum_loss, 3)}
                        """.strip()
                        )
                accum_loss = 0

        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        save_checkpoint(epoch, unwrapped_model, optimizer,
                        tokenizer, scheduler, ckpt_path)

        if accelerator.is_main_process:

            results_full, results_exp = sample_sequences(
                unwrapped_model, tokenizer, val_loader, limit=limit)

            resFileExp = caption_save_path + \
                'clevrx_captions_exp_' + str(epoch) + '.json'
            unf_resFileExp = caption_save_path + \
                'clevrx_unf_captions_exp_' + str(epoch) + '.json'
            unf_resFileFull = caption_save_path + \
                'clevrx_unf_captions_full_' + str(epoch) + '.json'
            save_scores_pathExp = caption_save_path + \
                'clevrx_scores_exp_' + str(epoch) + '.json'

            with open(unf_resFileExp, 'w') as w:
                print(f'write explanations to {unf_resFileExp}')
                json.dump(results_exp, w)

            with open(unf_resFileFull, 'w') as w:
                print(f'write full sequences to {unf_resFileFull}')
                json.dump(results_full, w)

            # unfiltered results
            # get_scores(annFileExp, unf_resFileExp, save_scores_pathExp)

            # filtered results
            filter_and_get_scores(
                resFileExp, save_scores_pathExp, results_full, results_exp)
