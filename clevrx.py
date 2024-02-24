import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as transforms
from transformers import GPT2Tokenizer, AutoConfig
from transformers import get_linear_schedule_with_warmup  # ,  AdamW
from torch.optim import AdamW
import json
from cococaption.pycocotools.coco import COCO
from cococaption.pycocoevalcap.eval import COCOEvalCap
from accelerate import Accelerator
from models.gpt import GPT2LMHeadModel
from models.clip_vit import ImageEncoder
from utils.eval_utils import top_filtering, ScoreTracker
from utils.clevrx import CLEVRXTrainDataset, CLEVRXEvalDataset
import argparse
import os.path as osp
import os
from os import getcwd
from tqdm import tqdm


def change_requires_grad(model, req_grad):
    for p in model.parameters():
        p.requires_grad = req_grad


# def load_checkpoint(ckpt_path, epoch, learning_rate):

#     model_name = 'nle_model_{}'.format(str(epoch))
#     tokenizer_name = 'nle_gpt2_tokenizer_0'
#     filename = 'ckpt_stats_' + str(epoch) + '.tar'

#     tokenizer = GPT2Tokenizer.from_pretrained(
#         ckpt_path + tokenizer_name)        # load tokenizer
#     model = GPT2LMHeadModel.from_pretrained(
#         ckpt_path + model_name).to(device)   # load model with config
#     opt = torch.load(ckpt_path + filename)
#     optimizer = get_optimizer(model, learning_rate)
#     optimizer.load_state_dict(opt['optimizer_state_dict'])
#     start_epoch = opt['epoch'] + 1
#     scheduler_dic = opt['scheduler']
#     del opt
#     torch.cuda.empty_cache()

#     return tokenizer, model, optimizer, scheduler_dic, start_epoch


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
        tokenizer, scheduler, args, **kwargs):
    
    epoch_str = str(epoch).rjust(2, '0')
    greyscale_str = '_greyscale' if args.greyscale else ''
    save_path = osp.join(args.ckpt_path, f'clevrx_ckpt_{epoch_str}{greyscale_str}')
    model_name = f'clevrx_nle_model_{epoch_str}{greyscale_str}'
    tokenizer_name = f'clevrx_nle_gpt2_tokenizer_{epoch_str}{greyscale_str}'
    filename = f'clevrx_ckpt_stats_{epoch_str}{greyscale_str}.tar'
    
    os.mkdir(save_path)

    if epoch == 0:
        tokenizer.save_pretrained(
            osp.join(save_path, tokenizer_name))   # save tokenizer

    unwrapped_model.save_pretrained(
        osp.join(save_path, model_name), save_function=accelerator.save)

    opt = {'epoch': epoch,
           'optimizer_state_dict': optimizer.state_dict(),
           'scheduler': scheduler.state_dict(),
           'args': args,
           **kwargs}

    print(f'save model checkpoint to {args.ckpt_path + filename}')

    accelerator.save(opt, osp.join(save_path, filename))


def filter_and_get_scores(
        nle_data_val_path, annFileExp,
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
    
    scores = cocoEval.eval

    with open(save_scores_pathExp, 'w') as w:
        json.dump(scores, w)
        
    return scores


def sample_sequences(model, image_encoder, tokenizer, loader, args, limit=None):

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
                logits = lm_logits[0, -1, :] / args.temperature
                logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
                probs = F.softmax(logits, dim=-1)
                prev = torch.topk(probs, 1)[
                    1] if args.no_sample else torch.multinomial(probs, 1)

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


def get_optimizer(model, learning_rate, weight_decay):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    return optimizer


def main(args):

    learning_rate = 2e-5 if not args.finetune_pretrained else 1e-5

    image_encoder = ImageEncoder(device).to(device)
    change_requires_grad(image_encoder, False)

    if args.load_from_epoch is not None:
        raise NotImplementedError
        # tokenizer, model, optimizer, scheduler_dic, start_epoch = load_checkpoint(
        #     args.ckpt_path, args.load_from_epoch, learning_rate)

    else:

        if args.finetune_pretrained:
            tokenizer, model = load_pretrained()
            optimizer = get_optimizer(model, learning_rate, args.weight_decay)
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
            config.img_size = args.img_size
            config.max_seq_len = args.max_seq_len
            config.add_cross_attention = True

            model = GPT2LMHeadModel.from_pretrained(
                'distilgpt2', config=config)
            model.resize_token_embeddings(len(tokenizer))
            model = model.to(device)
            optimizer = get_optimizer(model, learning_rate)
        start_epoch = 0

    print("Model Setup Ready...")

    transforms_list = [
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]),
    ]

    if args.greyscale:
        transforms_list.insert(
            1, transforms.Grayscale(num_output_channels=3)  # transform to grayscale
        )

    img_transform = transforms.Compose(transforms_list)

    print(f'convert images to greyscale: {args.greyscale}')
    print(f'transformations: {img_transform}')

    train_dataset = CLEVRXTrainDataset(path=args.nle_data_train_path,
                                       img_dir=args.img_dir,
                                       transform=img_transform,
                                       tokenizer=tokenizer,
                                       max_seq_len=args.max_seq_len)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               pin_memory=True)

    val_dataset = CLEVRXEvalDataset(path=args.nle_data_val_path,
                                    img_dir=args.img_dir,
                                    transform=img_transform,
                                    tokenizer=tokenizer,
                                    max_seq_len=args.max_seq_len)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=True)

    model, optimizer, train_loader = accelerator.prepare(
        model, optimizer, train_loader)

    score_tracker = ScoreTracker(stop_after_epochs=args.early_stopping_epochs)

    t_total = (len(train_loader) // args.gradient_accumulation_steps) * \
        args.num_train_epochs
    warmup_steps = 0   # 0.10 * t_total
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

    if args.load_from_epoch is not None:
        scheduler.load_state_dict(scheduler_dic)

    for epoch in range(start_epoch, args.num_train_epochs):

        model.train()
        accum_loss = 0

        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):

            if args.limit:
                if step == args.limit:
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
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            accum_loss += loss.item()

            if step % args.gradient_accumulation_steps == 0 or step == len(train_loader) - 1:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                accum_loss = 0

        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)

        if accelerator.is_main_process:

            results_full, results_exp = sample_sequences(
                unwrapped_model, image_encoder, tokenizer, val_loader, args, limit=args.limit)

            if not osp.isdir(args.caption_save_path):
                osp.mkdir(args.caption_save_path)

            resFileExp = args.caption_save_path + \
                'clevrx_captions_exp_' + str(epoch) + '.json'
            unf_resFileExp = args.caption_save_path + \
                'clevrx_unf_captions_exp_' + str(epoch) + '.json'
            unf_resFileFull = args.caption_save_path + \
                'clevrx_unf_captions_full_' + str(epoch) + '.json'
            save_scores_pathExp = args.caption_save_path + \
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
            scores = filter_and_get_scores(
                args.nle_data_val_path, args.annFileExp, resFileExp, save_scores_pathExp, results_full, results_exp)

            cider_score = scores['CIDEr']
            score_tracker(cider_score)
            score_tracker.print_summary()

        if args.save_every >= 1:
            save_model = (epoch % args.save_every == 0 or epoch == args.num_train_epochs - 1)
        else:
            save_model = score_tracker.counter == 0
            if not save_model:
                print('non maximum score -- do not save model weights')

        if save_model:
            print('save model')                
            save_checkpoint(epoch, unwrapped_model, optimizer,
                            tokenizer, scheduler, args)


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
    parser.add_argument('--print_step', default=500, type=int)
    parser.add_argument('--early_stopping_epochs', default=2, type=int)
    parser.add_argument('--save_every', default=-1, type=int)
    parser.add_argument('--greyscale', action='store_true')

    args = parser.parse_args()

    print('Train settings:')
    print(args)
    print(f'working directory: {getcwd()}')
    print(f'CUDA available: {torch.cuda.is_available()}')

    accelerator = Accelerator()
    device = accelerator.device

    print(f'Training on device: {device}')

    main(args)
