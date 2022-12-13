import os.path as osp
import os
import pickle
import json
import shutil
import argparse


def main(args):

    # read files
    print('reading input files')
    with open(osp.join(args.clevrx_location, 'train_images_ids_v0.7.10-recut.pkl'), 'rb') as f:
        train_images = pickle.load(f)

    with open(osp.join(args.clevrx_location, 'dev_images_ids_v0.7.10-recut.pkl'), 'rb') as f:
        dev_images = pickle.load(f)

    with open(osp.join(args.clevrx_location, 'CLEVR_train_explanations_v0.7.10.json')) as f:
        train_ann_file = json.load(f)

    # split train into train/dev

    clevr_train_anns = train_ann_file['questions']
    clevrx_train = []
    clevrx_dev = []

    print('splitting train into train/dev')
    for entry in clevr_train_anns:
        if entry['image_filename'] in train_images:
            clevrx_train.append(entry)
        elif entry['image_filename'] in dev_images:
            clevrx_dev.append(entry)
        else:
            raise Exception(
                f'Image {entry["image_filename"]} not found in train or dev sets!')

    assert len(
        {e['image_index'] for e in clevrx_train} & {
            e['image_index'] for e in clevrx_dev}
    ) == 0

    # save files

    for name, data in zip(['train', 'dev'], [clevrx_train, clevrx_dev]):
        info = train_ann_file['info']
        info['split'] = name
        out_data = {
            'info': info,
            'questions': data
        }
        fname = osp.join(args.out_dir, f'CLEVRX_{name}.json')
        print(f'write {name} annotations to {fname}')
        with open(fname, 'w') as f:
            json.dump(out_data, f)

    # store original val file as test split in output directory

    val_ann_file = osp.join(args.clevrx_location,
                            'CLEVR_val_explanations_v0.7.10.json')
    fname = fname = osp.join(args.out_dir, f'CLEVRX_test.json')

    print(f'copying annotations from {val_ann_file} to {fname}')
    shutil.copy(val_ann_file, fname)


if __name__ == '__main__':

    file_dir = osp.dirname(osp.abspath(__file__))
    project_dir = osp.join(file_dir, os.pardir)

    parser = argparse.ArgumentParser()
    parser.add_argument('--clevrx_location', required=True,
                        help='location of the CLEVR-X annotations')
    parser.add_argument('--out_dir',
                        default=osp.join(project_dir, 'data'),
                        help='output directory for final train/dev/test annotations')
    args = parser.parse_args()

    # normalize paths
    args.clevrx_location = osp.abspath(args.clevrx_location)
    args.out_dir = osp.abspath(args.out_dir)
    print(f'CLEVR-X location: {args.clevrx_location}')
    print(f'Output directory: {args.out_dir}')

    main(args)
