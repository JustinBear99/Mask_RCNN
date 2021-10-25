#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict
import torch
# import copy
# import json
# import numpy as np
# import random
# from PIL import Image, ImageDraw
# import cv2

from detectron2.data.datasets import register_coco_instances
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetMapper, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
import detectron2.data.transforms as T
# from detectron2.data import detection_utils as utils
# from detectron2.structures import BoxMode

# def randomDatasetdict():
#     image_list = os.listdir('./datasets/coco/annotations/train/JPEGImages')
#     random.shuffle(image_list)
#     image_name = image_list[0]
#     with open('./datasets/coco/annotations/train/annotations.json', 'r') as jsonfile:
#         annofile = json.load(jsonfile)
#     for it in annofile['images']:
#         if it['file_name'] == 'JPEGImages/' + image_name:
#             image_id = it['id']
#             height = it["height"]
#             width = it["width"]
#     annotations = []
#     for it in annofile['annotations']:
#         if it['image_id'] == image_id and it['category_id'] == 3:
#             annotations.append({'iscrowd': it["iscrowd"],
#                                 'bbox': it["bbox"],
#                                 'category_id': 3,
#                                 'segmentation': it["segmentation"],
#                                 'bbox_mode': BoxMode(1)})
#     dataset_dict = {'file_name': './datasets/coco/annotations/train/JPEGImages/' + image_name,
#                     'height': height,
#                     'width': width,
#                     'image_id': image_id,
#                     'annotations': annotations}
#     return dataset_dict

# def readInfo(dataset_dict):
#     image = utils.read_image(dataset_dict["file_name"], format="BGR")
#     height, width = dataset_dict["height"], dataset_dict["width"]
#     image = np.array(image)
#     masks = []
#     for annotation in dataset_dict["annotations"]:
#         points = annotation["segmentation"][0]
#         points_tuple = [ (points[i], points[i+1]) for ｉ in range(0, len(points)-1, 2) ]
#         img = Image.new('L', (width, height), 0)
#         ImageDraw.Draw(img).polygon(points_tuple, outline=1, fill=1)
#         mask = np.array(img)
#         mask = np.stack((mask,)*3, axis=-1)
#         masks.append(mask)
#     return image, masks
    
# def img_add(img, img2, masks2):
#     h,w, c = img.shape
    

# def LSJ(img, masks, min_scale=0.1, max_scale=2.0):
#     rescale_ratio = np.random.uniform(min_scale, max_scale)
#     h, w, _ = img.shape

#     # rescale
#     h_new, w_new = int(h * rescale_ratio), int(w * rescale_ratio)
#     img.resize((h_new, w_new))
#     masks = [ mask.resize((h_new, w_new)) for mask in masks ]

#     x, y = int(np.random.uniform(0, abs(w_new - w))), int(np.random.uniform(0, abs(h_new - h)))
#     if rescale_ratio <= 1.0:  # padding
#         img_pad = np.ones((h, w, 3), dtype=np.uint8) * 168
#         img_pad[y:y+h_new, x:x+w_new, :] = img
#         masks_pad = []
#         for mask in masks:
#             mask_pad = np.zeros((h, w), dtype=np.uint8)
#             mask_pad[y:y+h_new, x:x+w_new] = mask
#             masks_pad.append(mask_pad)
#     else:  # crop
#         img_crop = img[y:y+h, x:x+w, :]
#         masks_crop = [ mask[y:y+h, x:x+w] for mask in masks ]
#     return img_crop, masks_crop
    

# def mapper(dataset_dict):
    
#     dataset_dict = copy.deepcopy(dataset_dict)
#     # image, masks = readInfo(dataset_dict)
#     image = utils.read_image(dataset_dict["file_name"], format="BGR")
#     image = np.array(image)
#     dataset_dict_2 = randomDatasetdict()
#     image_2 = utils.read_image(dataset_dict_2["file_name"], format="BGR")
    
#     height = dataset_dict_2["height"]
#     width = dataset_dict_2["width"]
#     for annotation in dataset_dict_2["annotations"]:
#         points = annotation["segmentation"][0]
#         [x, y, w, h] = annotation["bbox"]
#         img = Image.new('L', (width, height), 0)
#         points_tuple = [ (points[i], points[i+1]) for ｉ in range(0, len(points)-1, 2) ]
#         # print(points_tuple)
#         ImageDraw.Draw(img).polygon(points_tuple, outline=1, fill=1)
#         mask_2 = np.array(img)
#         mask_2 = np.stack((mask_2,)*3, axis=-1)




#         cropped_img = image_2 * mask_2
#         # im = Image.fromarray(cropped_img)
#         # im.save("cropped_img.jpg")
#         cropped_img = cropped_img[int(y):int(y+h), int(x):int(x+w), :]
#         # im = Image.fromarray(cropped_img)
#         # im.save("cropped_img_small.jpg")
#         cropped_segmentation = [ point-w if index%2==0 else point-h for index, point in enumerate(points) ]
#         new_x = random.randint(int((width-w)*1/4), int((width-w)*3/4))
#         new_y = random.randint(int((height-h)*1/4), int((height-h)*3/4))
#         # print('height:', height, 'width:', width, x, y, w, h,'new_x:', new_x, 'new_y:', new_y, 'new_x+w:', new_x+w, 'new_y+h:', new_y+h)
#         new_segmentation = [ point+new_x if index%2==0 else point+new_y for index, point in enumerate(cropped_segmentation) ]

        

#         for i in range(new_y, new_y + int(h)-1):
#             for j in range(new_x, new_x + int(w)-1):
#                 for k in range(0, 2):
#                     if cropped_img[i-new_y, j-new_x, k] == 0:
#                         pass
#                     else:
#                         image[i, j, k] = cropped_img[i-new_y, j-new_x, k]
#         dataset_dict['annotations'].append({'iscrowd': 0,
#                                             'bbox': [new_x, new_y, w, h],
#                                             'category_id': 3,
#                                             'segmentation': [new_segmentation],
#                                             'bbox_mode': BoxMode(1)
#         })
#     # print(dataset_dict)
#     # auginput = T.AugInput(image)
#     # transform = T.Resize((1920, 1080))(auginput)
#     # im = Image.fromarray(image)
#     # im.save("cropped_img_final.jpg")

#     image, transform = T.apply_transform_gens([T.Resize((800, 800))], image)
#     # image = torch.from_numpy(image.transpose(2, 0, 1))
#     image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
#     annos = [
#         utils.transform_instance_annotations(annotation, transform, image.shape[:2])
#         for annotation in dataset_dict.pop("annotations")
#     ]
#     return {
#        "image": image,
#        "width": width,
#        "height": height,
#        "instances": utils.annotations_to_instances(annos, image.shape[:2])
#     }


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop. You can use
    "tools/plain_train_net.py" as an example.
    """
    @classmethod
    def build_train_loader(cls, cfg):
        # return build_detection_train_loader(cfg, mapper=mapper)
        return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, is_train=True, augmentations=[
            T.ResizeShortestEdge(short_edge_length=[900, 1200], max_size=1600, sample_style='range'),
           T.RandomFlip(),
            # T.RandomRotation(-30, 30)
        #     T.RandomCrop('relative_range', (0.5, 0.5)),
        #     T.RandomContrast(0.8, 1.2),
        #     T.RandomBrightness(0.8, 1.2),
        #     T.RandomExtent((0.2, 0.8), (0.2, 0.2))
        ]))

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    # register_coco_instances('asparagus_train', {'_background_': 0, 'clump': 1, 'stalk': 2, 'spear': 3, 'bar': 4} , "./datasets/coco/annotations/train/annotations.json", "./datasets/coco/annotations/train")
    # register_coco_instances('asparagus_val', {'_background_': 0, 'clump': 1, 'stalk': 2, 'spear': 3, 'bar': 4} , "./datasets/coco/annotations/test_458/annotations.json", "./datasets/coco/annotations/test_458")
    register_coco_instances('asparagus_train', {'_background_': 0, 'clump': 1, 'stalk': 2, 'spear': 3, 'bar': 4, 'straw': 5} , "./datasets/coco/annotations/straw/train/annotations.json", "./datasets/coco/annotations/straw/train")
    register_coco_instances('asparagus_val', {'_background_': 0, 'clump': 1, 'stalk': 2, 'spear': 3, 'bar': 4, 'straw': 5} , "./datasets/coco/annotations/val_straw/val/annotations.json", "./datasets/coco/annotations/val_straw/val")
    
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
