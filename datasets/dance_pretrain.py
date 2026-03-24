# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
MOT dataset which returns image_id for evaluation.
"""
from collections import defaultdict
import json
import os
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.utils.data
import os.path as osp
from PIL import Image, ImageDraw
import copy
import datasets.transforms as T
from models.structures import Instances
import random
from random import choice, randint
from PIL import ImageFilter
from torchvision.transforms import transforms
def get_random_patch_from_img(img, min_pixel=8):
    """
    :param img: original image
    :param min_pixel: min pixels of the query patch
    :return: query_patch,x,y,w,h
    """
    w, h = img.size
    min_w, max_w = min_pixel, w - min_pixel
    min_h, max_h = min_pixel, h - min_pixel
    sw, sh = np.random.randint(min_w, max_w + 1), np.random.randint(min_h, max_h + 1)
    x, y = np.random.randint(w - sw) if sw != w else 0, np.random.randint(h - sh) if sh != h else 0
    patch = img.crop((x, y, x + sw, y + sh))
    return patch, x, y, sw, sh

def get_warped_patch_from_img(
    img,
    x, y, sw, sh,
    max_shift_ratio=0.05,
    max_scale_ratio=0.05
):
    """
    Warp patch slightly for temporal consistency pretraining.
    """
    w, h = img.size
    x = int(x)
    y = int(y)
    sw = int(sw)
    sh = int(sh)
    # shift
    max_dx = int(sw * max_shift_ratio)
    max_dy = int(sh * max_shift_ratio)
    dx = np.random.randint(-max_dx, max_dx + 1)
    dy = np.random.randint(-max_dy, max_dy + 1)

    # scale
    scale = 1.0 + np.random.uniform(-max_scale_ratio, max_scale_ratio)
    new_sw = int(sw * scale)
    new_sh = int(sh * scale)

    # new position
    new_x = np.clip(x + dx, 0, w - new_sw)
    new_y = np.clip(y + dy, 0, h - new_sh)
    # import pdb;pdb.set_trace()
    patch = img.crop((new_x, new_y, new_x + new_sw, new_y + new_sh))
    
    return patch, new_x, new_y, new_sw, new_sh

class DetMOTDetection:
    def __init__(self, args, data_txt_path: str, seqs_folder, dataset2transform, query_transform):
        self.args = args
        self.dataset2transform = dataset2transform
        self.query_transform = query_transform
        self.num_frames_per_batch = max(args.sampler_lengths)
        self.sample_mode = args.sample_mode
        self.sample_interval = args.sample_interval
        self.video_dict = {}
        self.split_dir = os.path.join(args.mot_path, "DanceTrack",  "images","train")
        self.frame_interval = args.frame_interval
        # import pdb;pdb.set_trace()
        self.labels_full = defaultdict(lambda : defaultdict(list))
        for vid in os.listdir(self.split_dir):
            if 'DPM' in vid or 'FRCNN' in vid:
                print(f'filter {vid}')
                continue
            # import pdb;pdb.set_trace()
            if self.frame_interval == 0 or self.frame_interval == None:
                gt_path = os.path.join(self.split_dir, vid, 'gt', 'gt.txt')
            else:
                sort_file = 'gt_sort_' + str(self.frame_interval) + '.txt'
                gt_path = os.path.join(self.split_dir, vid, 'gt', sort_file)
            for l in open(gt_path):
                t, i, *xywh, mark, label = l.strip().split(',')[:8]
                t, i, mark, label = map(int, (t, i, mark, label))
                if mark == 0:
                    continue
                if label in [3, 4, 5, 6, 9, 10, 11]:  # Non-person
                    continue
                else:
                    crowd = False
                x, y, w, h = map(float, (xywh))
                self.labels_full[vid][t].append([x, y, w, h, i, crowd])
        vid_files = list(self.labels_full.keys())

        self.indices = []
        self.vid_tmax = {}
        for vid in vid_files:
            self.video_dict[vid] = len(self.video_dict)
            t_min = min(self.labels_full[vid].keys())
            t_max = max(self.labels_full[vid].keys()) + 1
            self.vid_tmax[vid] = t_max - 1
            for t in range(t_min, t_max - self.num_frames_per_batch):
                self.indices.append((vid, t))

        self.sampler_steps: list = args.sampler_steps
        self.lengths: list = args.sampler_lengths
        print("sampler_steps={} lenghts={}".format(self.sampler_steps, self.lengths))
        self.period_idx = 0

        self.prev_pseudo_boxes = {}

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        if self.sampler_steps is None or len(self.sampler_steps) == 0:
            # fixed sampling length.
            return

        for i in range(len(self.sampler_steps)):
            if epoch >= self.sampler_steps[i]:
                self.period_idx = i + 1
        print("set epoch: epoch {} period_idx={}".format(epoch, self.period_idx))
        self.num_frames_per_batch = self.lengths[self.period_idx]

    def step_epoch(self):
        # one epoch finishes.
        print("Dataset: epoch {} finishes".format(self.current_epoch))
        self.set_epoch(self.current_epoch + 1)

    @staticmethod
    def _targets_to_instances(targets: dict, img_shape) -> Instances:
        gt_instances = Instances(tuple(img_shape))
        gt_instances.boxes = targets['boxes']
        gt_instances.labels = targets['labels']
        gt_instances.obj_ids = targets['obj_ids']
        gt_instances.area = targets['area']
        # gt_instances.patches = targets['patches']
        return gt_instances

    def load_crowd(self):
        path, boxes, crowd = choice(self.crowd_gts)
        img = Image.open(path)

        w, h = img._size
        boxes = torch.tensor(boxes, dtype=torch.float32)
        areas = boxes[..., 2:].prod(-1)
        boxes[:, 2:] += boxes[:, :2]
        target = {
            'boxes': boxes,
            'labels': torch.zeros((len(boxes), ), dtype=torch.long),
            'iscrowd': torch.as_tensor(crowd),
            'image_id': torch.tensor([0]),
            'area': areas,
            'obj_ids': torch.arange(len(boxes)),
            'size': torch.as_tensor([h, w]),
            'orig_size': torch.as_tensor([h, w]),
            'dataset': "CrowdHuman",
        }
        return [img], [target]

    def _pre_single_frame(self, vid, idx: int):
        # import pdb;pdb.set_trace()
        if self.frame_interval != 0 and self.frame_interval != None:
            img_idx = (idx-1)*self.frame_interval + 1
            img_path = os.path.join(self.split_dir, vid, 'img1', f'{img_idx:08d}.jpg')
        else:
            img_path = os.path.join(self.split_dir, vid, 'img1', f'{idx:08d}.jpg')
        # img_path = os.path.join(self.split_dir, vid, 'img1', f'{idx:08d}.jpg')
        img = Image.open(img_path)
        targets = {}
        w, h = img._size
        assert w > 0 and h > 0, "invalid image {} with shape {} {}".format(img_path, w, h)
        obj_idx_offset = self.video_dict[vid] * 100000  # 100000 unique ids is enough for a video.

        targets['dataset'] = 'MOT17'
        targets['boxes'] = []
        targets['area'] = []
        targets['iscrowd'] = []
        targets['labels'] = []
        targets['obj_ids'] = []
        targets['image_id'] = torch.as_tensor(idx)
        targets['size'] = torch.as_tensor([h, w])
        targets['orig_size'] = torch.as_tensor([h, w])

        
        prev_boxes = self.prev_pseudo_boxes.get(vid, None)

        if prev_boxes is not None: # 非第一帧
            # pass
            boxes = []
            patches = []
            for (x, y, sw, sh) in self.prev_pseudo_boxes[vid]:
                sw -= x
                sh -= y
                patch, nx, ny, sw, sh = get_warped_patch_from_img(
                    img, x, y, sw, sh
                )
                patches.append(self.query_transform(patch))
                boxes.append([nx, ny, nx + sw, ny + sh])

            boxes = torch.tensor(boxes, dtype=torch.float32)
            self.prev_pseudo_boxes[vid] = boxes.clone()
        else: # 第一帧
            # pass
            num_pseudo = self.args.num_patches
            boxes = []
            patches = []

            while len(boxes) < num_pseudo:
                patch, x, y, sw, sh = get_random_patch_from_img(img)
                boxes.append([x, y, x + sw, y + sh])
                patches.append(self.query_transform(patch))

            boxes = torch.tensor(boxes, dtype=torch.float32)
            self.prev_pseudo_boxes[vid] = boxes.clone()

        targets['boxes'] = boxes  # pseudo GT
        targets['labels'] = torch.ones(len(boxes), dtype=torch.long)
        targets['obj_ids'] = torch.arange(len(boxes)) + obj_idx_offset  # 伪 ID，只用于 matching
        targets['iscrowd'] = torch.zeros(len(boxes), dtype=torch.bool)
        targets['area'] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        # targets['patches'] = torch.stack(patches, dim=0)

        # for *xywh, id, crowd in self.labels_full[vid][idx]:
        #     targets['boxes'].append(xywh)
        #     targets['area'].append(xywh[2] * xywh[3])
        #     targets['iscrowd'].append(crowd)
        #     targets['labels'].append(0)
        #     targets['obj_ids'].append(id + obj_idx_offset)
        # import pdb;pdb.set_trace()
        targets['area'] = torch.as_tensor(targets['area'])
        targets['iscrowd'] = torch.as_tensor(targets['iscrowd'])
        targets['labels'] = torch.as_tensor(targets['labels'])
        targets['obj_ids'] = torch.as_tensor(targets['obj_ids'], dtype=torch.float64)
        targets['boxes'] = torch.as_tensor(targets['boxes'], dtype=torch.float32).reshape(-1, 4)
        # targets['boxes'][:, 2:] += targets['boxes'][:, :2] #cxcywh to xyxy
        # import pdb;pdb.set_trace()
        return img, targets, torch.stack(patches, dim=0)

    def _get_sample_range(self, start_idx):

        # take default sampling method for normal dataset.
        assert self.sample_mode in ['fixed_interval', 'random_interval'], 'invalid sample mode: {}'.format(self.sample_mode)
        if self.sample_mode == 'fixed_interval':
            sample_interval = self.sample_interval
        elif self.sample_mode == 'random_interval':
            sample_interval = np.random.randint(1, self.sample_interval + 1)
        default_range = start_idx, start_idx + (self.num_frames_per_batch - 1) * sample_interval + 1, sample_interval
        return default_range

    def pre_continuous_frames(self, vid, indices):
        return zip(*[self._pre_single_frame(vid, i) for i in indices])

    def sample_indices(self, vid, f_index):
        assert self.sample_mode == 'random_interval'
        rate = randint(1, self.sample_interval + 1)
        tmax = self.vid_tmax[vid]
        ids = [f_index + rate * i for i in range(self.num_frames_per_batch)]
        return [min(i, tmax) for i in ids]

    def __getitem__(self, idx):
        vid, f_index = self.indices[idx]
        indices = self.sample_indices(vid, f_index)
        images, targets, patches = self.pre_continuous_frames(vid, indices)
        dataset_name = targets[0]['dataset']
        transform = self.dataset2transform[dataset_name]
        # import pdb;pdb.set_trace()
        if transform is not None:
            images, targets = transform(images, targets)
        # from models.visualizer import ImageVisualizer
        # visualizer = ImageVisualizer(denormalized=True)
        # # import pdb;pdb.set_trace()
        # visualizer.show(images, targets,box_format='cxcywh')
        # import pdb;pdb.set_trace()
        gt_instances = []
        for img_i, targets_i in zip(images, targets):
            gt_instances_i = self._targets_to_instances(targets_i, img_i.shape[1:3])
            gt_instances.append(gt_instances_i)
        # import pdb;pdb.set_trace()
        # from models.visualizer import ImageVisualizer
        # visualizer = ImageVisualizer(denormalized=True)
        # import pdb;pdb.set_trace()
        # visualizer.show(list(patches[0]), box_format='cxcywh')
        # visualizer.show(images,gt_instances,box_format='cxcywh')
        # visualizer.show(images)
        # import pdb;pdb.set_trace()
        return {
            'imgs': images,
            'gt_instances': gt_instances,
            'patches': list(patches),  #长度为clip，每一个元素是形状为n,3,w,h的tensor，n为当前帧伪gt box的个数，wh为patch的尺寸
        }

    def __len__(self):
        return len(self.indices)


class DetMOTDetectionValidation(DetMOTDetection):
    def __init__(self, args, seqs_folder, dataset2transform):
        args.data_txt_path = args.val_data_txt_path
        super().__init__(args, seqs_folder, dataset2transform)


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def get_query_transforms(image_set):
    if image_set == 'train':
        # SimCLR style augmentation
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.ToTensor(),
            # transforms.RandomHorizontalFlip(),  HorizontalFlip may cause the pretext too difficult, so we remove it
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    if image_set == 'val':
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    raise ValueError(f'unknown {image_set}')

def make_transforms_for_mot17(image_set, args=None):

    normalize = T.MotCompose([
        T.MotToTensor(),
        T.MotNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    scales = [608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992]

    if image_set == 'train':
        return T.MotCompose([
            T.MotRandomResize(scales, max_size=600),
            # T.MotRandomHorizontalFlip(),
            # T.MotRandomSelect(
            #     T.MotRandomResize(scales, max_size=1536),
            #     T.MotCompose([
            #         T.MotRandomResize([800, 1000, 1200]),
            #         T.FixedMotRandomCrop(800, 1200),
            #         T.MotRandomResize(scales, max_size=1536),
            #     ])
            # ),
            normalize,
        ])

    if image_set == 'val':
        return T.MotCompose([
            T.MotRandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build_dataset2transform(args, image_set):
    mot17_train = make_transforms_for_mot17('train', args)
    mot17_test = make_transforms_for_mot17('val', args)

    dataset2transform_train = {'MOT17': mot17_train}
    dataset2transform_val = {'MOT17': mot17_test}
    if image_set == 'train':
        return dataset2transform_train
    elif image_set == 'val':
        return dataset2transform_val
    else:
        raise NotImplementedError()



def build(image_set, args):
    root = Path(args.mot_path)
    assert root.exists(), f'provided MOT path {root} does not exist'
    dataset2transform = build_dataset2transform(args, image_set)
    query_transform = get_query_transforms(image_set)
    if image_set == 'train':
        data_txt_path = args.data_txt_path_train
        dataset = DetMOTDetection(args, data_txt_path=data_txt_path, seqs_folder=root, dataset2transform=dataset2transform,
                                  query_transform=query_transform)
    if image_set == 'val':
        data_txt_path = args.data_txt_path_val
        dataset = DetMOTDetection(args, data_txt_path=data_txt_path, seqs_folder=root, dataset2transform=dataset2transform,
                                  query_transform=query_transform)
    return dataset
