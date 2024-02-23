import json
import pickle
import os.path as osp
import os
import argparse
from data_utils import cocoann_file_template


def main(args):

    ##################
    # Prepare Splits #
    ##################

    files_exist = all(
        [
            f in os.listdir(args.clevrx_location)
            for f in ["CLEVRX_train.json", "CLEVRX_dev.json", "CLEVRX_test.json"]
        ]
    )

    if args.overwrite_existing_files or not files_exist:

        print("creating CLEVR-X splits")

        print("reading data...")
        traindev_path = osp.join(
            args.clevrx_location, "CLEVR_train_explanations_v0.7.10.json"
        )
        with open(traindev_path) as f:
            data = json.load(f)
            traindev_questions = data["questions"]

        test_path = osp.join(
            args.clevrx_location, "CLEVR_val_explanations_v0.7.10.json"
        )
        with open(test_path) as f:
            data = json.load(f)
            test_questions = data["questions"]

        train_imgs_path = osp.join(
            args.clevrx_location, "train_images_ids_v0.7.10-recut.pkl"
        )
        with open(train_imgs_path, "rb") as f:
            train_imgs = pickle.load(f)

        dev_imgs_path = osp.join(
            args.clevrx_location, "dev_images_ids_v0.7.10-recut.pkl"
        )
        with open(dev_imgs_path, "rb") as f:
            dev_imgs = pickle.load(f)

        print("separating train and dev data...")
        train_questions = [
            q for q in traindev_questions if q["image_filename"] in train_imgs
        ]
        dev_questions = [
            q for q in traindev_questions if q["image_filename"] in dev_imgs
        ]

        assert len(train_questions) + len(dev_questions) == len(traindev_questions)
        assert (
            len(
                {t["image_filename"] for t in train_questions}
                & {d["image_filename"] for d in dev_questions}
            )
            == 0
        )

        train_outfile = osp.join(args.clevrx_location, "CLEVRX_train.json")
        print(f"write train data to {train_outfile}")
        with open(train_outfile, "w") as f:
            json.dump(train_questions, f)

        dev_outfile = osp.join(args.clevrx_location, "CLEVRX_dev.json")
        print(f"write dev data to {dev_outfile}")
        with open(dev_outfile, "w") as f:
            json.dump(dev_questions, f)

        test_outfile = osp.join(args.clevrx_location, "CLEVRX_test.json")
        print(f"write test data to {test_outfile}")
        with open(test_outfile, "w") as f:
            json.dump(test_questions, f)

    else:
        print("CLEVR-X splits exist -- skipping...")

    #####################
    # Prepare nle Files #
    #####################

    files_exist = all(
        [
            f"clevrX_{split}.json" in os.listdir(args.nle_path)
            for split in ["train", "dev", "test"]
        ]
    )

    if args.overwrite_existing_files or not files_exist:

        for split in ["train", "dev", "test"]:
            print(f"{split} split")
            clevrx_anns = osp.join(args.clevrx_location, f"CLEVRX_{split}.json")
            print(f"input file: {clevrx_anns}")

            with open(clevrx_anns) as f:
                clevrx_data = json.load(f)

            id2data = dict()

            for entry in clevrx_data:
                qid = str(entry["question_index"])

                id2data[qid] = dict()

                id2data[qid]["question"] = entry["question"]
                id2data[qid]["answer"] = entry["answer"]
                id2data[qid]["image_id"] = str(entry["image_index"])
                id2data[qid]["image_name"] = entry["image_filename"]
                id2data[qid]["explanation"] = [
                    e.lower().replace(".", "") for e in entry["factual_explanation"]
                ]

            out_file = osp.join(args.nle_path, f"clevrX_{split}.json")
            print(f"write to {out_file}")
            with open(out_file, "w") as w:
                json.dump(id2data, w)

    else:
        print("nle files exist -- skipping...")

    #############################
    # Prepare cococaption files #
    #############################

    cocoann_path = osp.join(base_path, "cococaption", "annotations")
    if not osp.isdir(cocoann_path):
        raise Exception(
            f"path {cocoann_path} does not exist -- init & update submodules first!"
        )

    files_exist = all(
        [
            f"clevrX_{split}_annot_{condition}.json" in os.listdir(args.nle_path)
            for split in ["train", "dev", "test"]
            for condition in ["exp", "full"]
        ]
    )

    if args.overwrite_existing_files or not files_exist:

        for split in ["train", "dev", "test"]:

            in_path = os.path.join(args.nle_path, f"clevrX_{split}.json")
            out_path = os.path.join(cocoann_path, f"clevrX_{split}_annot_exp.json")
            data = json.load(open(in_path, "r"))

            out = cocoann_file_template
            out.update({"images": [], "annotations": []})

            cnt = 0

            for qid, qid_data in data.items():

                out["images"].append({"id": int(qid)})

                for s in qid_data["explanation"]:

                    if len(s) == 0:
                        print("Warning: {} has no annotations".format(qid))
                        continue

                    out["annotations"].append(
                        {"image_id": out["images"][-1]["id"], "caption": s, "id": cnt}
                    )
                    cnt += 1

            json.dump(out, open(out_path, "w"))
            print("wrote to ", out_path)

            # Explanations + Answers

            in_path = os.path.join(args.nle_path, f"clevrX_{split}.json")
            out_path = os.path.join(cocoann_path, f"clevrX_{split}_annot_full.json")

            data = json.load(open(in_path, "r"))

            out = cocoann_file_template
            out.update({"images": [], "annotations": []})

            cnt = 0

            for qid, qid_data in data.items():

                out["images"].append({"id": int(qid)})
                for s in qid_data["explanation"]:

                    if len(s) == 0:
                        print("Warning: {} has no annotations".format(qid))
                        continue

                    s = qid_data["answer"] + " because " + s
                    out["annotations"].append(
                        {"image_id": out["images"][-1]["id"], "caption": s, "id": cnt}
                    )
                    cnt += 1

            json.dump(out, open(out_path, "w"))
            print("wrote to ", out_path)

    else:
        print("cococaption annotations exist -- skipping...")


if __name__ == "__main__":

    file_path = osp.dirname(osp.realpath(__file__))
    base_path = osp.abspath(osp.join(file_path, os.pardir))

    parser = argparse.ArgumentParser()
    parser.add_argument("--clevrx_location", required=True)
    parser.add_argument(
        "--nle_path",
        default=osp.join(base_path, "nle_data", "CLEVR-X"),
    )
    parser.add_argument("--overwrite_existing_files", action="store_true")

    args = parser.parse_args()

    args.base_path = base_path

    if not osp.isdir(args.nle_path):
        print(f"create output directory {args.nle_path}...")
        os.mkdir(args.nle_path)

    main(args)
