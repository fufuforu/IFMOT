# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

import torch.utils.data

from .coco import build as build_coco
from .dancetrack import build as build_ifmot_dancetrack
from .dancetrack_pretrain import build as build_ifmot_dancetrack_pretrain
from .mot17 import build as build_ifmot_mot17
from .mot17_pretrain import build as build_ifmot_mot17_pretrain
from .torchvision_datasets import CocoDetection


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    builders = {
        'coco': build_coco,
        'ifmot_mot17_pretrain': build_ifmot_mot17_pretrain,
        'ifmot_mot17': build_ifmot_mot17,
        'ifmot_dancetrack_pretrain': build_ifmot_dancetrack_pretrain,
        'ifmot_dancetrack': build_ifmot_dancetrack,
    }

    if args.dataset_file == 'coco_panoptic':
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)

    if args.dataset_file not in builders:
        raise ValueError(f'dataset {args.dataset_file} not supported')
    return builders[args.dataset_file](image_set, args)
