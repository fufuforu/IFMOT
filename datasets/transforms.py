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
Transforms and data augmentation for both image + bbox.
"""
import copy
import random
import PIL
import torch
import torchvision
import torchvision.transforms as T
import torchvision.transforms.v2 as Tv2
import torchvision.transforms.functional as F
import torchvision.transforms.v2.functional as Fv2
from PIL import Image, ImageDraw
from util.box_ops import box_xyxy_to_cxcywh
from util.misc import interpolate
import numpy as np
import os 
from typing import Any, Dict, List, Optional, Callable, Union, cast
from torchvision.transforms.v2.utils import query_bounding_box, query_spatial_size
from torch.utils._pytree import tree_flatten, tree_unflatten
from torchvision import datapoints


def crop_mot(image, target, region):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd", "obj_ids"]

    if "boxes" in target:
        boxes = target["boxes"]
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            max_size = torch.as_tensor([w, h], dtype=torch.float32)
            cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
            cropped_boxes = cropped_boxes.clamp(min=0)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target


def random_shift(image, target, region, sizes):
    oh, ow = sizes
    # step 1, shift crop and re-scale image firstly
    cropped_image = F.crop(image, *region)
    cropped_image = F.resize(cropped_image, sizes)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd", "obj_ids"]

    if "boxes" in target:
        boxes = target["boxes"]
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes *= torch.as_tensor([ow / w, oh / h, ow / w, oh / h])
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            max_size = torch.as_tensor([w, h], dtype=torch.float32)
            cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
            cropped_boxes = cropped_boxes.clamp(min=0)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target


def crop(image, target, region):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]
    if 'obj_ids' in target:
        fields.append('obj_ids')

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)

        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target


def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    return flipped_image, target


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target


def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class MotRandomCrop(RandomCrop):
    def __call__(self, imgs: list, targets: list):
        ret_imgs = []
        ret_targets = []
        region = T.RandomCrop.get_params(imgs[0], self.size)
        for img_i, targets_i in zip(imgs, targets):
            img_i, targets_i = crop(img_i, targets_i, region)
            ret_imgs.append(img_i)
            ret_targets.append(targets_i)
        return ret_imgs, ret_targets

class FixedMotRandomCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, imgs: list, targets: list):
        ret_imgs = []
        ret_targets = []
        w = random.randint(self.min_size, min(imgs[0].width, self.max_size))
        h = random.randint(self.min_size, min(imgs[0].height, self.max_size))
        region = T.RandomCrop.get_params(imgs[0], [h, w])
        for img_i, targets_i in zip(imgs, targets):
            img_i, targets_i = crop_mot(img_i, targets_i, region)
            ret_imgs.append(img_i)
            ret_targets.append(targets_i)
        return ret_imgs, ret_targets

class MotRandomShift(object):
    def __init__(self, bs=1):
        self.bs = bs

    def __call__(self, imgs: list, targets: list):
        ret_imgs = copy.deepcopy(imgs)
        ret_targets = copy.deepcopy(targets)

        n_frames = len(imgs)
        select_i = random.choice(list(range(n_frames)))
        w, h = imgs[select_i].size

        xshift = (100 * torch.rand(self.bs)).int()
        xshift *= (torch.randn(self.bs) > 0.0).int() * 2 - 1 
        yshift = (100 * torch.rand(self.bs)).int()
        yshift *= (torch.randn(self.bs) > 0.0).int() * 2 - 1
        ymin = max(0, -yshift[0])
        ymax = min(h, h - yshift[0])
        xmin = max(0, -xshift[0])
        xmax = min(w, w - xshift[0])

        region = (int(ymin), int(xmin), int(ymax-ymin), int(xmax-xmin))
        ret_imgs[select_i], ret_targets[select_i] = random_shift(imgs[select_i], targets[select_i], region, (h,w)) 
        
        return ret_imgs, ret_targets


class FixedMotRandomShift(object):
    def __init__(self, bs=1, padding=50):
        self.bs = bs
        self.padding = padding

    def __call__(self, imgs: list, targets: list):
        ret_imgs = []
        ret_targets = []

        n_frames = len(imgs)
        w, h = imgs[0].size
        xshift = (self.padding * torch.rand(self.bs)).int() + 1
        xshift *= (torch.randn(self.bs) > 0.0).int() * 2 - 1
        yshift = (self.padding * torch.rand(self.bs)).int() + 1
        yshift *= (torch.randn(self.bs) > 0.0).int() * 2 - 1
        ret_imgs.append(imgs[0])
        ret_targets.append(targets[0])
        for i in range(1, n_frames):
            ymin = max(0, -yshift[0])
            ymax = min(h, h - yshift[0])
            xmin = max(0, -xshift[0])
            xmax = min(w, w - xshift[0])
            prev_img = ret_imgs[i-1].copy()
            prev_target = copy.deepcopy(ret_targets[i-1])
            region = (int(ymin), int(xmin), int(ymax - ymin), int(xmax - xmin))
            img_i, target_i = random_shift(prev_img, prev_target, region, (h, w))
            ret_imgs.append(img_i)
            ret_targets.append(target_i)

        return ret_imgs, ret_targets


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)


class MotRandomSizeCrop(RandomSizeCrop):
    def __call__(self, imgs, targets):
        w = random.randint(self.min_size, min(imgs[0].width, self.max_size))
        h = random.randint(self.min_size, min(imgs[0].height, self.max_size))
        region = T.RandomCrop.get_params(imgs[0], [h, w])
        ret_imgs = []
        ret_targets = []
        for img_i, targets_i in zip(imgs, targets):
            img_i, targets_i = crop(img_i, targets_i, region)
            ret_imgs.append(img_i)
            ret_targets.append(targets_i)
        return ret_imgs, ret_targets


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class MotCenterCrop(CenterCrop):
    def __call__(self, imgs, targets):
        image_width, image_height = imgs[0].size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        ret_imgs = []
        ret_targets = []
        for img_i, targets_i in zip(imgs, targets):
            img_i, targets_i = crop(img_i, targets_i, (crop_top, crop_left, crop_height, crop_width))
            ret_imgs.append(img_i)
            ret_targets.append(targets_i)
        return ret_imgs, ret_targets


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class MotRandomHorizontalFlip(RandomHorizontalFlip):
    def __call__(self, imgs, targets):
        if random.random() < self.p:
            ret_imgs = []
            ret_targets = []
            for img_i, targets_i in zip(imgs, targets):
                img_i, targets_i = hflip(img_i, targets_i)
                ret_imgs.append(img_i)
                ret_targets.append(targets_i)
            return ret_imgs, ret_targets
        return imgs, targets


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class MotRandomResize(RandomResize):
    def __call__(self, imgs, targets):
        size = random.choice(self.sizes)
        ret_imgs = []
        ret_targets = []
        for img_i, targets_i in zip(imgs, targets):
            img_i, targets_i = resize(img_i, targets_i, size, self.max_size)
            ret_imgs.append(img_i)
            ret_targets.append(targets_i)
        return ret_imgs, ret_targets


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class MotRandomPad(RandomPad):
    def __call__(self, imgs, targets):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        ret_imgs = []
        ret_targets = []
        for img_i, targets_i in zip(imgs, targets):
            img_i, target_i = pad(img_i, targets_i, (pad_x, pad_y))
            ret_imgs.append(img_i)
            ret_targets.append(targets_i)
        return ret_imgs, ret_targets


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class MotRandomSelect(RandomSelect):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __call__(self, imgs, targets):
        if random.random() < self.p:
            return self.transforms1(imgs, targets)
        return self.transforms2(imgs, targets)


class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target


class MotToTensor(ToTensor):
    def __call__(self, imgs, targets):
        ret_imgs = []
        for img in imgs:
            ret_imgs.append(F.to_tensor(img))
        return ret_imgs, targets


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class MotRandomErasing(RandomErasing):
    def __call__(self, imgs, targets):
        # TODO: Rewrite this part to ensure the data augmentation is same to each image.
        ret_imgs = []
        for img_i, targets_i in zip(imgs, targets):
            ret_imgs.append(self.eraser(img_i))
        return ret_imgs, targets


class MoTColorJitter(T.ColorJitter):
    def __call__(self, imgs, targets):
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        ret_imgs = []
        for img_i, targets_i in zip(imgs, targets):
            ret_imgs.append(transform(img_i))
        return ret_imgs, targets


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        if target is not None:
            target['ori_img'] = image.clone()
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return image, target


class MotNormalize(Normalize):
    def __call__(self, imgs, targets=None):
        ret_imgs = []
        ret_targets = []
        for i in range(len(imgs)):
            img_i = imgs[i]
            targets_i = targets[i] if targets is not None else None
            img_i, targets_i = super().__call__(img_i, targets_i)
            ret_imgs.append(img_i)
            ret_targets.append(targets_i)
        return ret_imgs, ret_targets


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class MotCompose(Compose):
    def __call__(self, imgs, targets):
        for t in self.transforms:
            imgs, targets = t(imgs, targets)
        return imgs, targets

class RandomZoomOut(Tv2.RandomZoomOut):

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        orig_h, orig_w = query_spatial_size(flat_inputs)

        r = self.side_range[0] + torch.rand(1) * (self.side_range[1] - self.side_range[0])
        canvas_width = int(orig_w * r)
        canvas_height = int(orig_h * r)

        r = torch.rand(2)
        left = int((canvas_width - orig_w) * r[0])
        top = int((canvas_height - orig_h) * r[1])
        right = canvas_width - (left + orig_w)
        bottom = canvas_height - (top + orig_h)
        padding = [left, top, right, bottom]

        params = {
            'padding': padding,
            'needs_pad': torch.rand(1) >= self.p
        }

        return params

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:

        if not params['needs_pad']:
            return inpt
        else:
            params_ = copy.deepcopy(params)
            params_.pop('needs_pad')
            return super()._transform(inpt, params_)

    def forward(self, *inputs: Any) -> Any:
        # We need to almost duplicate `Transform.forward()` here since we always want to check the inputs, but return
        # early afterwards in case the random check triggers. The same result could be achieved by calling
        # `super().forward()` after the random check, but that would call `self._check_inputs` twice.

        inputs = inputs if len(inputs) > 1 else inputs[0]
        flat_inputs, spec = tree_flatten(inputs)

        self._check_inputs(flat_inputs)

        needs_transform_list = self._needs_transform_list(flat_inputs)
        params = self._get_params(
            [inpt for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list) if needs_transform]
        )

        flat_outputs = [
            self._transform(inpt, params) if needs_transform else inpt
            for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list)
        ]

        return tree_unflatten(flat_outputs, spec)


class RandomPerspective(Tv2.RandomPerspective):
    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        params = super()._get_params(flat_inputs)
        params['needs_perspective'] = torch.rand(1) >= self.p
        return params
    
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:

        if not params['needs_perspective']:
            return inpt
        else:
            params_ = copy.deepcopy(params)
            params_.pop('needs_perspective')
            return super()._transform(inpt, params_)

    def forward(self, *inputs: Any) -> Any:
        # We need to almost duplicate `Transform.forward()` here since we always want to check the inputs, but return
        # early afterwards in case the random check triggers. The same result could be achieved by calling
        # `super().forward()` after the random check, but that would call `self._check_inputs` twice.

        inputs = inputs if len(inputs) > 1 else inputs[0]
        flat_inputs, spec = tree_flatten(inputs)

        self._check_inputs(flat_inputs)

        needs_transform_list = self._needs_transform_list(flat_inputs)
        params = self._get_params(
            [inpt for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list) if needs_transform]
        )

        flat_outputs = [
            self._transform(inpt, params) if needs_transform else inpt
            for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list)
        ]

        return tree_unflatten(flat_outputs, spec)

class RandomIoUCrop(Tv2.RandomIoUCrop):
    def __init__(self, min_scale: float = 0.3, max_scale: float = 1, min_aspect_ratio: float = 0.5, max_aspect_ratio: float = 2, sampler_options: Optional[List[float]] = None, trials: int = 40, p: float = 1.0):
        super().__init__(min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, sampler_options, trials)
        self.p = p 

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        params = {}
        params['needs_crop'] = torch.rand(1) >= self.p # RandomCrop also has the key 'needs_crop'
        if params['needs_crop']:
            p = super()._get_params(flat_inputs)
            params.update(p)
            if len(p) == 0: # all trials are failed
                params['needs_crop'] = False

        return params

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:

        if not params['needs_crop']:
            return inpt
        else:
            return super()._transform(inpt, params)

class SanitizeBoundingBox(Tv2.SanitizeBoundingBox):
    def __init__(self, *args, labels_getter=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._labels_getter = labels_getter
    def forward(self, *inputs: Any) -> Any:
        #import pdb; pdb.set_trace()
        inputs = inputs if len(inputs) > 1 else inputs[0]
        # import pdb; pdb.set_trace()
        if self._labels_getter is None:
            labels = None
        else:
            labels = self._labels_getter(inputs)
        
        # import pdb; pdb.set_trace()
        if labels is not None:
            msg = "The labels in the input to forward() must be a tensor or None, got {type} instead."
            if isinstance(labels, torch.Tensor):
                labels = (labels,)
            elif isinstance(labels, (tuple, list)):
                for entry in labels:
                    if not isinstance(entry, torch.Tensor):
                        # TODO: we don't need to enforce tensors, just that entries are indexable as t[bool_mask]
                        raise ValueError(msg.format(type=type(entry)))
            else:
                raise ValueError(msg.format(type=type(labels)))

        flat_inputs, spec = tree_flatten(inputs)
        #import pdb; pdb.set_trace()
        # TODO: this enforces one single BoundingBox entry.
        # Assuming this transform needs to be called at the end of *any* pipeline that has bboxes...
        # should we just enforce it for all transforms?? What are the benefits of *not* enforcing this?
        boxes = query_bounding_box(flat_inputs)

        if boxes.ndim != 2:
            raise ValueError(f"boxes must be of shape (num_boxes, 4), got {boxes.shape}")

        if labels is not None:
            for label in labels:
                if boxes.shape[0] != label.shape[0]:
                    raise ValueError(
                        f"Number of boxes (shape={boxes.shape}) and must match the number of labels."
                        f"Found labels with shape={label.shape})."
                    )

        boxes = cast(
            datapoints.BoundingBox,
            Fv2.convert_format_bounding_box(
                boxes,
                new_format=datapoints.BoundingBoxFormat.XYXY,
            ),
        )
        ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
        valid = (ws >= self.min_size) & (hs >= self.min_size) & (boxes >= 0).all(dim=-1)
        # TODO: Do we really need to check for out of bounds here? All
        # transforms should be clamping anyway, so this should never happen?
        image_h, image_w = boxes.spatial_size
        valid &= (boxes[:, 0] <= image_w) & (boxes[:, 2] <= image_w)
        valid &= (boxes[:, 1] <= image_h) & (boxes[:, 3] <= image_h)

        params = dict(valid=valid, labels=labels)  
        flat_outputs = [
            # Even-though it may look like we're transforming all inputs, we don't:
            # _transform() will only care about BoundingBoxes and the labels
            self._transform(inpt, params)
            for inpt in flat_inputs
        ]
        outputs = tree_unflatten(flat_outputs, spec)

        return outputs

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        is_label = params["labels"] is not None and any(inpt is label for label in params["labels"])
        is_bounding_box_or_mask = isinstance(inpt, (datapoints.BoundingBox, datapoints.Mask))

        if not (is_label or is_bounding_box_or_mask):
            return inpt

        output = inpt[params["valid"]]

        if is_label:
            return output

        return type(inpt).wrap_like(inpt, output)

class RandomHorizontalFlipv2(Tv2.RandomHorizontalFlip):
    
    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        params = {}
        params['needs_flip'] = torch.rand(1) >= self.p
        return params
    
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:

        if not params['needs_flip']:
            return inpt
        else:
            return super()._transform(inpt, params)

    def forward(self, *inputs: Any) -> Any:
        # We need to almost duplicate `Transform.forward()` here since we always want to check the inputs, but return
        # early afterwards in case the random check triggers. The same result could be achieved by calling
        # `super().forward()` after the random check, but that would call `self._check_inputs` twice.

        inputs = inputs if len(inputs) > 1 else inputs[0]
        flat_inputs, spec = tree_flatten(inputs)

        self._check_inputs(flat_inputs)

        needs_transform_list = self._needs_transform_list(flat_inputs)
        params = self._get_params(
            [inpt for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list) if needs_transform]
        )

        flat_outputs = [
            self._transform(inpt, params) if needs_transform else inpt
            for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list)
        ]

        return tree_unflatten(flat_outputs, spec)

class ConvertBox(Tv2.Transform):
    _transformed_types = (
        datapoints.BoundingBox,
    )
    def __init__(self, out_fmt='', normalize=False) -> None:
        super().__init__()
        self.out_fmt = out_fmt
        self.normalize = normalize

        self.data_fmt = {
            'xyxy': datapoints.BoundingBoxFormat.XYXY,
            'cxcywh': datapoints.BoundingBoxFormat.CXCYWH
        }

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:  
        if self.out_fmt:
            spatial_size = inpt.spatial_size
            in_fmt = inpt.format.value.lower()
            inpt = torchvision.ops.box_convert(inpt, in_fmt=in_fmt, out_fmt=self.out_fmt)
            inpt = datapoints.BoundingBox(inpt, format=self.data_fmt[self.out_fmt], spatial_size=spatial_size)
        
        if self.normalize:
            inpt = inpt / torch.tensor(inpt.spatial_size[::-1]).tile(2)[None]

        return inpt

def labels_getter_func_for_mot_in_SanitizeBoundingBox(inputs):
    targets = inputs[1] # image, targets
    labels = []
    # import pdb;pdb.set_trace()
    for k in ['labels', 'obj_ids', 'area', 'iscrowd']:
        labels.append(targets[k])
    return tuple(labels)

def mot_transform_wrap(transform_class, **args):
    class MOTTransformWrapper(transform_class):
        def __init__(self, **args):
            super().__init__(**args)
            self.__class__.__name__ = 'MOT_{}'.format(transform_class.__name__) #TODO: '{}_{}'.format('MOTTransformWrapper', transform_class.__name__)
            self.parent_class_name = transform_class.__name__
            self._params_ = None
        
        def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
            
            if self._params_ is None:
                self._params_ = super()._get_params(flat_inputs)

            params = copy.deepcopy(self._params_)

            if self.parent_class_name in ['RandomIoUCrop']:
                # import pdb; pdb.set_trace()
                if params['needs_crop']: # we need to check if one box is within the image
                    bboxes = flat_inputs[1]
                    assert isinstance(bboxes, (datapoints.BoundingBox,)), 'Not Boxes!'
                    xyxy_bboxes = Fv2.convert_format_bounding_box(
                                    bboxes.as_subclass(torch.Tensor), bboxes.format, datapoints.BoundingBoxFormat.XYXY
                                )
                    cx = 0.5 * (xyxy_bboxes[..., 0] + xyxy_bboxes[..., 2])
                    cy = 0.5 * (xyxy_bboxes[..., 1] + xyxy_bboxes[..., 3])
                    left, new_h, top, new_w = params['left'], params['height'], params['top'], params['width'], 
                    right = left + new_w
                    bottom = top + new_h
                    is_within_crop_area = (left < cx) & (cx < right) & (top < cy) & (cy < bottom)
                    params['is_within_crop_area'] = is_within_crop_area
                    
            return params
        
        def forward(self, *inputs: Any) -> Any:

            # if self.parent_class_name in ['SanitizeBoundingBox']:
                # import pdb; pdb.set_trace()
            self._params_ = None
            inputs = inputs if len(inputs) > 1 else inputs[0]
            outputs = []
            #import pdb; pdb.set_trace()
            for inp in inputs:
                
                output = super().forward(inp)
                outputs.append(output)

            return outputs

    return MOTTransformWrapper(**args)

class Transforms_Image(object):
    def __init__(self, bs=1):
        self.bs = bs

    def __call__(self, imgs: list, targets: list):
        video = []
        for i in range(self.bs):
            video.append((imgs[0], targets[0]))
        #import pdb;pdb.set_trace()
        randomaffine = Tv2.RandomAffine(degrees=1, translate=[0.05, 0.05], fill=255)
        resize = Tv2.Resize(size=[640, 640], antialias=False)
        for i in range(self.bs):
            video[i] = randomaffine(video[i])
            video[i] = resize(video[i])
        return video
        
    
class Transforms_Video(Transforms_Image):
    def __call__(self, video: list):
        
        #randomzoomout = RandomZoomOut(fill=255)
        randomzoomout = mot_transform_wrap(RandomZoomOut,fill=255)
        randomperspective = mot_transform_wrap(RandomPerspective,distortion_scale=0.5, p=0.5, fill=255)
        randomaffine = mot_transform_wrap(Tv2.RandomAffine,degrees=10, translate=[0.3, 0.3], scale=[0.75, 1.5],fill=255)

        randomchoice = Tv2.RandomChoice(
            transforms=[
                randomzoomout,
                randomperspective,
                randomaffine
            ],
            p=[0.5, 0.25, 0.25]  # 必须加起来等于1
        )
        video = randomchoice(video)

        randomioucrop = mot_transform_wrap(RandomIoUCrop,
                                            # min_scale=0.1,
                                            # max_scale=0.5,
                                            # min_aspect_ratio=0.3,
                                            # max_aspect_ratio=3.3,
                                            # sampler_options=[0.0, 0.1, 0.3],  # 允许低 IOU，可能完全丢失目标
                                            p=0.8)
        sanitizeboundingbox = mot_transform_wrap(SanitizeBoundingBox,min_size=1,labels_getter=labels_getter_func_for_mot_in_SanitizeBoundingBox)
        randomhorizontalfilpv2 = mot_transform_wrap(RandomHorizontalFlipv2)
        resize = mot_transform_wrap(Tv2.Resize,size=[640, 640], antialias=False)

        #import pdb;pdb.set_trace()
        video = randomioucrop(video)
        video = sanitizeboundingbox(video)
        video = randomhorizontalfilpv2(video)
        video = resize(video)
        #import pdb;pdb.set_trace()
        #video = mottotensor(video)
        # video = normalize(video)
        # video = sanitizeboundingbox(video)
        # video = convertbox(video)
        #import pdb;pdb.set_trace()
        images, targets = zip(*video)
        return list(images), list(targets)
    