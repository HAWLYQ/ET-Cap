# Copyright (c) Facebook, Inc. and its affiliates.

# The following script requires Java 1.8.0 and pycocotools installed.
# The pycocoevalcap can be installed with pip as
# pip install git+https://github.com/flauted/coco-caption.git@python23
# Original pycocoevalcap code is at https://github.com/tylin/coco-caption
# but has no python3 support yet.

import json
import argparse
# from builtins import dict
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice


class COCOEvalCap:
    """
    COCOEvalCap code is adopted from https://github.com/tylin/coco-caption
    """

    def __init__(self, img_ids, coco, coco_res, metrics):
        self.eval_imgs = []
        self.eval = dict()
        self.img_to_eval = dict()
        self.coco = coco
        self.coco_res = coco_res
        self.metrics = metrics

    def evaluate(self):
        gts = self.coco
        res = self.coco_res

        # =================================================
        # Set up scorers
        # =================================================
        # print("tokenization...")
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)
        
        # print('datasets/coco_caption_eval.py:', len(gts.keys()), len(res.keys()))
        try:
            assert gts.keys() == res.keys()
        except AssertionError as e:
            print('diff key(in gt):', set(list(gts.keys()))-set(list(res.keys())))
            print('diff key(in pred):', set(list(res.keys()))-set(list(gts.keys())))
            exit(0)

        # =================================================
        # Set up scorers
        # =================================================
        # print("setting up scorers...")
        scorers = [
            ('BLEU', Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            ('METEOR', Meteor(), "METEOR"),
            ("ROUGE_L", Rouge(), "ROUGE_L"),
            ("CIDEr", Cider(), "CIDEr"),
            ("SPICE", Spice(), "SPICE"),
        ]

        # =================================================
        # Compute scores
        # =================================================
        for name, scorer, method in scorers:
            # print("computing %s score..." % (scorer.method()))
            if name in self.metrics:
                score, scores = scorer.compute_score(gts, res)
                if type(method) == list:
                    for sc, scs, m in zip(score, scores, method):
                        self.set_eval(sc, m)
                        self.set_img_to_eval_imgs(scs, gts.keys(), m)
                        # print("%s: %0.3f" % (m, sc))
                else:
                    self.set_eval(score, method)
                    self.set_img_to_eval_imgs(scores, gts.keys(), method)
                    # print("%s: %0.3f" % (method, score))
        self.set_eval_imgs()
        #  anwen hu 2020/9/16
        """for img_id in res.keys():
            # print('res_id', res_id)
            hypo = res[img_id]
            gt_captions = gts[img_id]
            cider = self.img_to_eval[img_id]['CIDEr']
            if cider*100 < 20:
                print(img_id, cider, hypo)
                print(gt_captions)
                print('=================')"""

    def set_eval(self, score, method):
        self.eval[method] = score

    def set_img_to_eval_imgs(self, scores, img_ids, method):
        for img_id, score in zip(img_ids, scores):
            if img_id not in self.img_to_eval:
                self.img_to_eval[img_id] = dict()
                self.img_to_eval[img_id]["image_id"] = img_id
            self.img_to_eval[img_id][method] = score


    def set_eval_imgs(self):
        self.eval_imgs = [eval for img_id, eval in self.img_to_eval.items()]


def calculate_metrics(img_ids, dataset_dts, dataset_res, metrics):
    img_to_anns_gts = {id: [] for id in img_ids}
    for ann in dataset_dts["annotations"]:
        img_to_anns_gts[ann["image_id"]] += [ann]

    img_to_anns_res = {id: [] for id in img_ids}
    for ann in dataset_res["annotations"]:
        img_to_anns_res[ann["image_id"]] += [ann]

    # print('datasets/coco_caption_eval.py:', len(img_to_anns_gts.keys()), len(img_to_anns_res.keys()))
    try:
        assert img_to_anns_gts.keys() == img_to_anns_res.keys()
    except AssertionError as e:
        print('diff key(in gt):', set(list(img_to_anns_gts.keys()))-set(list(img_to_anns_res.keys())))
        print('diff key(in pred):', set(list(img_to_anns_res.keys()))-set(list(img_to_anns_gts.keys())))
        exit(0)

    eval_obj = COCOEvalCap(img_ids, img_to_anns_gts, img_to_anns_res, metrics)
    eval_obj.evaluate()
    return eval_obj.eval, eval_obj.img_to_eval


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image captioning metrics")
    parser.add_argument("--reference_json", help="Path to reference captions json")
    parser.add_argument("--predicted_json", help="Path to predicted captions json")
    args = parser.parse_args()

    with open(args.reference_json, "r") as f:
        captions = json.load(f)

    references = []
    img_ids = []

    for img in captions["images"]:
        if img["split"] == "test":
            for c in img["sentences"]:
                d = {}
                d["image_id"] = c["imgid"]
                img_ids.append(c["imgid"])
                d["caption"] = c["raw"]
                references.append(d)
    img_ids = list(set(img_ids))

    with open(args.predicted_json, "r") as f:
        preds = json.load(f)

    dataset_dts = {"annotations": references}
    dataset_res = {"annotations": preds}
    print(calculate_metrics(img_ids, dataset_dts, dataset_res))
