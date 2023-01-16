import torch
import torch.utils.data
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from transformers import GPT2Tokenizer
import json
from PIL import Image
from models.gpt import GPT2LMHeadModel
from models.clip_vit import ImageEncoder
from utils import data_utils
import argparse
import os
import os.path as osp
from utils.eval_utils import top_filtering
import re
from tqdm import tqdm


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

        img_path = osp.join(self.img_dir, split, img_name)
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        qid = torch.LongTensor([int(question_id)])

        return (img, qid, input_ids, segment_ids)

    def __len__(self):
        return len(self.ids_list)


def change_requires_grad(model, req_grad):
    for p in model.parameters():
        p.requires_grad = req_grad

    
def load_trained(ckpt_path, epoch, device):

    model_name = 'nle_model_{}'.format(str(epoch))
    tokenizer_name = 'nle_gpt2_tokenizer_0'
    filename = 'ckpt_stats_' + str(epoch) + '.tar'

    tokenizer = GPT2Tokenizer.from_pretrained(
        osp.join(ckpt_path, tokenizer_name))        # load tokenizer
    model = GPT2LMHeadModel.from_pretrained(
        osp.join(ckpt_path, model_name)).to(device)   # load model with config

    return tokenizer, model


def sample_sequences(model, image_encoder, tokenizer, loader, args):

    model.eval()
    results_exp = []
    results_full = []
    SPECIAL_TOKENS = ['<|endoftext|>', '<pad>',
                      '<question>', '<answer>', '<explanation>']
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    because_token = tokenizer.convert_tokens_to_ids('Ġbecause')

    for i, batch in tqdm(enumerate(loader), total=len(loader)):

        current_output = []
        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        img, img_id, input_ids, segment_ids = batch
        img_embeddings = image_encoder(img)
        always_exp = False

        with torch.no_grad():

            for step in range(args.max_seq_len + 1):

                if step == args.max_seq_len:
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
                logits = top_filtering(
                    logits, top_k=args.top_k, top_p=args.top_p)
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

                new_segment = torch.LongTensor([new_segment]).to(args.device)
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

    return results_full, results_exp


def main(args):

    image_encoder = ImageEncoder(args.device).to(args.device)
    change_requires_grad(image_encoder, False)

    if not osp.isdir(args.caption_save_path):
        os.mkdir(args.caption_save_path)

    tokenizer, model = load_trained(
        args.ckpt_path, args.load_from_epoch, args.device)
    print("Model Setup Ready...")

    if args.greyscale: 
        print('sampling from greyscale images')
        img_transform = transforms.Compose([transforms.Resize((args.img_size, args.img_size)),
                                            transforms.Grayscale(num_output_channels=3), # transform to grayscale
                                            transforms.ToTensor(),
                                            transforms.Normalize(
                                                mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
                                            ])

    else: 
        print('sampling from color images')
        img_transform = transforms.Compose([transforms.Resize((args.img_size, args.img_size)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(
                                                mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
                                            ])

    annot_path = args.nle_data_test_path if args.split == 'test' else args.nle_data_val_path
    print(f'annotation path: {annot_path}')

    test_dataset = CLEVRXEvalDataset(path=annot_path,
                                   transform=img_transform,
                                   tokenizer=tokenizer,
                                   max_seq_len=args.max_seq_len,
                                   img_dir=args.image_dir)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.eval_batch_size,
                                              shuffle=False,
                                              pin_memory=True)

    results_full, results_exp = sample_sequences(
        model, image_encoder, tokenizer, test_loader, args)

    colour_settings = '_greyscale' if args.greyscale else ''
    unf_resFileExp = f'clevrx_unf_captions_exp_{str(args.load_from_epoch)}_{args.split}{colour_settings}.json'
    unf_resFileFull = f'clevrx_unf_captions_full_{str(args.load_from_epoch)}_{args.split}{colour_settings}.json'

    with open(osp.join(args.caption_save_path, unf_resFileExp), 'w') as w:
        json.dump(results_exp, w)

    with open(osp.join(args.caption_save_path, unf_resFileFull), 'w') as w:
        json.dump(results_full, w)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_batch_size',
                        default=1, type=int)
    parser.add_argument('--img_size',
                        default=224, type=int)
    parser.add_argument('--image_dir',
                        default='images/')
    parser.add_argument('--ckpt_path',
                        default='ckpts/CLEVR-X/')
    parser.add_argument('--caption_save_path',
                        default='generated/')
    parser.add_argument('--nle_data_test_path',
                        default='nle_data/CLEVR-X/clevrX_test.json')
    parser.add_argument('--nle_data_val_path',
                        default='nle_data/CLEVR-X/clevrX_dev.json')
    parser.add_argument('--max_seq_len',
                        default=20, type=int)
    parser.add_argument('--load_from_epoch',
                        default=11, type=int)
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu', choices=['cuda', 'cpu'])
    parser.add_argument('--no_sample',
                        default=True, type=bool)
    parser.add_argument('--top_k',
                        default=0, type=int)
    parser.add_argument('--top_p',
                        default=0.9, type=float)
    parser.add_argument('--temperature',
                        default=1, type=int)
    parser.add_argument('--split', 
                        default='val', choices=['val', 'test'])
    parser.add_argument('--greyscale',
                        action='store_true', help='convert images into greyscale')

    args = parser.parse_args()

    print(args)

    main(args)
