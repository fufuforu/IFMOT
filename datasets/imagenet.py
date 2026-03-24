import os
import json
import random
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import datasets.transforms as T
from torchvision.transforms import transforms
from PIL import ImageFilter
from models.structures import Instances


import torch.nn.functional as F

def sim(patches_list):
    # 假设
    # patches_list[0]: [9, 3, 128, 128]
    # patches_list[1]: [9, 3, 128, 128]

    p0 = patches_list[0].view(9, -1)  # [9, 49152]
    p1 = patches_list[1].view(9, -1)  # [9, 49152]

    # L2 归一化
    A_normalized = F.normalize(p0, p=2, dim=1)
    B_normalized = F.normalize(p1, p=2, dim=1)
    # import pdb;pdb.set_trace()
    # 计算余弦相似度
    cosine_similarity = torch.mm(A_normalized, B_normalized.t())
    return cosine_similarity

def get_patch_from_img(img, boxes):
    """
    :param img: original image
    :param min_pixel: min pixels of the query patch
    :return: query_patch,x,y,w,h
    """
    # import pdb;pdb.set_trace()
    x1, y1, x2, y2 = boxes[0].tolist()
    patch = img.crop((x1, y1, x2, y2))
    return patch

class ImageNetPseudoVideoDataset(Dataset):
    """
    Generate pseudo videos from ImageNet detection annotations (COCO-style).
    Each video is composed of repeated static images with shuffled temporal order.
    """

    def __init__(self, args, data_txt_path: str, seqs_folder, dataset2transform, query_transform):
        self.args = args
        self.dataset2transform = dataset2transform
        self.img_root = args.img_root
        self.ann_file = args.ann_file
        self.transforms = self.dataset2transform
        self.query_transform = query_transform

        self.num_pictures_row = args.num_pictures_row
        self.num_pictures_column = args.num_pictures_column 
        self.pseudo_num_images = self.num_pictures_row * self.num_pictures_column
        # self.pseudo_repeat = args.pseudo_repeat
        self.pseudo_shuffle = args.pseudo_shuffle
        # self.pseudo_max_boxes = args.pseudo_max_boxes

        # ---- load annotation ----
        with open(self.ann_file, "r") as f:
            coco = json.load(f)

        self.images = coco["images"]
        self.annotations = coco["annotations"]

        # image_id -> image info
        self.id2img = {img["id"]: img for img in self.images}
        # self.image_ids = list(self.id2img.keys())

        # annotation_id -> annotations
        self.imgid2anns = defaultdict(list)
        for ann in self.annotations:
            self.imgid2anns[ann["id"]].append(ann)
        self.annotation_ids = list(self.imgid2anns.keys())

        print(
            f"[ImageNetPseudoVideo] Loaded {len(self.images)} images, "
            f"{len(self.annotations)} annotations"
        )
        # import pdb;pdb.set_trace()
        self.sampler_steps: list = args.sampler_steps
        self.lengths: list = args.sampler_lengths
        print("sampler_steps={} lenghts={}".format(self.sampler_steps, self.lengths))
        self.period_idx = 0

        self.progress = 0.0
        self.num_frames_per_batch = self.compute_video_len(self.progress)
        # self.print_flag = True


    def set_progress(self, progress):
        self.progress = progress
        self.num_frames_per_batch = self.compute_video_len(progress)

    def compute_video_len(self, progress):
        """
        progress: float in [0, 1]
        """
        assert 0.0 <= progress <= 1.0

        n = len(self.lengths)
        # progress ∈ [0,1] → index ∈ [0, n-1]
        idx = int(progress * n)

        # 防止 progress == 1.0 时越界
        idx = min(idx, n - 1)
        print("when steps={} lenghts={}".format(idx, self.lengths[idx]))
        return self.lengths[idx]
            
    

    # def set_epoch(self, epoch):
    #     self.current_epoch = epoch
    #     if self.sampler_steps is None or len(self.sampler_steps) == 0:
    #         # fixed sampling length.
    #         return

    #     for i in range(len(self.sampler_steps)):
    #         if epoch >= self.sampler_steps[i]:
    #             self.period_idx = i + 1
    #     print("set epoch: epoch {} period_idx={}".format(epoch, self.period_idx))
    #     self.num_frames_per_batch = self.lengths[self.period_idx]

    # def step_epoch(self):
    #     # one epoch finishes.
    #     print("Dataset: epoch {} finishes".format(self.current_epoch))
    #     self.set_epoch(self.current_epoch + 1)

    @staticmethod
    def _targets_to_instances(targets: dict, img_shape) -> Instances:
        gt_instances = Instances(tuple(img_shape))
        gt_instances.boxes = targets['boxes']
        gt_instances.labels = targets['labels']
        gt_instances.obj_ids = targets['obj_ids']
        gt_instances.area = targets['area']
        # gt_instances.patches = targets['patches']
        return gt_instances

    def __len__(self):
        # length is virtual; one sample = one pseudo video
        # return len(self.annotation_ids)
        return len(self.annotation_ids)//2

    # ------------------------------------------------
    # core logic
    # ------------------------------------------------
    def _sample_pseudo_video(self):
        """sample image ids for one pseudo video"""
        annotation_ids = random.sample(self.annotation_ids, self.pseudo_num_images)

        # frame_img_ids = []
        # for img_id in img_ids:
        #     frame_img_ids.extend([img_id] * self.pseudo_repeat)

        if self.pseudo_shuffle:
            random.shuffle(annotation_ids)

        return annotation_ids


    def scale_boxes(self, boxes, orig_size, new_size):
        """
        boxes: Tensor[N,4] in xyxy
        orig_size: (w, h)
        new_size: (w, h)
        """
        ow, oh = orig_size
        nw, nh = new_size
        scale_w = nw / ow
        scale_h = nh / oh
        boxes[:, 0] *= scale_w
        boxes[:, 2] *= scale_w
        boxes[:, 1] *= scale_h
        boxes[:, 3] *= scale_h
        return boxes
    def make_grid_pseudo_video(self, images, targets, n_rows, n_cols, shuffle_times):


        n_imgs = len(images)
        assert n_imgs <= n_rows * n_cols, "图片数量不能超过 grid 大小"

        # pad images if not enough
        if n_imgs < n_rows * n_cols:
            images = images + [images[-1]] * (n_rows * n_cols - n_imgs)
            targets = targets + [targets[-1]] * (n_rows * n_cols - n_imgs)

        # 生成帧
        new_images = []
        new_targets = []
        patches_list = []
        for _ in range(shuffle_times):
        # 随机打乱
            indices = list(range(len(images)))
            random.shuffle(indices)
            shuffled_images = [images[i] for i in indices]
            shuffled_targets = [targets[i] for i in indices]

            # 计算列最大宽度和行最大高度
            widths, heights = zip(*[img.size for img in shuffled_images])
            col_max_widths = [max(widths[i::n_cols]) for i in range(n_cols)]
            row_max_heights = [max(heights[i*n_cols:(i+1)*n_cols]) for i in range(n_rows)]

            total_width = sum(col_max_widths)
            total_height = sum(row_max_heights)

            canvas = Image.new("RGB", (total_width, total_height))

            y_offset = 0
            all_boxes, all_labels, all_areas, all_iscrowd, all_obj_ids = [], [], [], [], []
            next_obj_id = 0  # 为整帧生成连续 obj_ids

            patches = []
            
            for row in range(n_rows):
                x_offset = 0
                for col in range(n_cols):
                    idx = row * n_cols + col
                    img = shuffled_images[idx]
                    tgt = shuffled_targets[idx]
                    w, h = img.size
                    canvas.paste(img, (x_offset, y_offset))

                    # 偏移boxes
                    boxes = tgt["boxes"].clone()
                    boxes[:, 0] += x_offset
                    boxes[:, 1] += y_offset
                    boxes[:, 2] += x_offset
                    boxes[:, 3] += y_offset

                    n_obj = boxes.shape[0]

                    all_boxes.append(boxes)
                    all_labels.append(tgt["labels"].clone())
                    all_areas.append(tgt["area"].clone())
                    all_iscrowd.append(tgt["iscrowd"].clone())
                    all_obj_ids.append(tgt["obj_ids"].clone())
                    next_obj_id += n_obj
                    # import pdb;pdb.set_trace()
                    patch = get_patch_from_img(img, tgt["boxes"])
                    patches.append(self.query_transform(patch))

                    x_offset += col_max_widths[col]
                y_offset += row_max_heights[row]

            # 合并所有小图targets
            merged_target = {
                "boxes": torch.cat(all_boxes, dim=0),
                "labels": torch.cat(all_labels, dim=0),
                "area": torch.cat(all_areas, dim=0),
                "iscrowd": torch.cat(all_iscrowd, dim=0),
                "obj_ids": torch.cat(all_obj_ids, dim=0),
                "image_id": torch.tensor([0]),
                "size": torch.tensor([total_height, total_width]),
                "orig_size": torch.tensor([total_height, total_width]),
                "dataset": "ImageNetPseudo",
            }
            patches_list.append(torch.stack(patches, dim=0))
            new_images.append(canvas)
            new_targets.append(merged_target)

        return new_images, new_targets, patches_list

    def __getitem__(self, idx):
        
        annotation_ids = self._sample_pseudo_video()

        images = []
        targets = []

        # stable pseudo IDs per image
        annid2objid = {}
        next_obj_id = 0

        for annotation_id in annotation_ids:
            anns = self.imgid2anns[annotation_id]
            img_id = anns[0]['image_id']
            img_info = self.id2img[img_id]
            img_path = os.path.join(self.img_root, img_info["file_name"])

            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            # import pdb;pdb.set_trace()
            # anns = self.imgid2anns[img_id]

            # assign pseudo IDs (once per image)
            if annotation_id not in annid2objid:
                annid2objid[annotation_id] = next_obj_id
                next_obj_id += 1

            obj_ids_this_img = [annid2objid[annotation_id]]
            boxes, labels, areas, obj_ids, iscrowd = [], [], [], [], []

            for ann, obj_id in zip(anns, obj_ids_this_img):
                x, y, bw, bh = ann["bbox"]
                boxes.append([x, y, x + bw, y + bh])
                areas.append(ann.get("area", bw * bh))
                labels.append(1)  # single foreground class
                iscrowd.append(ann.get("iscrowd", 0))
                obj_ids.append(obj_id)

            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64),
                "area": torch.tensor(areas, dtype=torch.float32),
                "iscrowd": torch.tensor(iscrowd, dtype=torch.int64),
                "obj_ids": torch.tensor(obj_ids, dtype=torch.int64),
                "image_id": torch.tensor([img_id]),
                "size": torch.tensor([h, w]),
                "orig_size": torch.tensor([h, w]),
                "dataset": "ImageNetPseudo",
            }

            images.append(img)
            targets.append(target)
        # import pdb;pdb.set_trace()
        # # 原始 9 张
        # images_9 = images
        # targets_9 = targets

        # 第一次拼接
        grid_img, grid_tgt, patches_list = self.make_grid_pseudo_video(images, targets, self.num_pictures_row,self.num_pictures_column, self.num_frames_per_batch)
        # import pdb;pdb.set_trace()
        # # 打乱顺序
        # perm = torch.randperm(9).tolist()
        # images_perm = [images_9[i] for i in perm]
        # targets_perm = [targets_9[i] for i in perm]

        # # 第二次拼接
        # grid_img_B, grid_tgt_B = self.make_grid(images_perm, targets_perm)

        # 组成伪视频（2 帧
        images = grid_img
        targets = grid_tgt

        if self.transforms is not None:
            images, targets = self.transforms(images, targets)

        gt_instances = []
        for img_i, targets_i in zip(images, targets):
            gt_instances_i = self._targets_to_instances(targets_i, img_i.shape[1:3])
            gt_instances.append(gt_instances_i)

        # from models.visualizer import ImageVisualizer
        # visualizer = ImageVisualizer(denormalized=True)
        # import pdb;pdb.set_trace()
        # visualizer.show(list(patches[0]), box_format='cxcywh')
        # visualizer.show(images,gt_instances,box_format='cxcywh')
        # visualizer.show(images)
        
        # sim_matrix = sim(patches_list)
        # import pdb;pdb.set_trace()
        return {
            "imgs": images,      # list[T] of Tensor
            "gt_instances": gt_instances,  # list[T] of dict
            'patches': patches_list,
            'patches_ids': [target['obj_ids'] for target in targets]
        }

class ImageNetPseudoVideoDatasetValidation(ImageNetPseudoVideoDataset):
    def __init__(self, args, seqs_folder, dataset2transform):
        args.data_txt_path = args.val_data_txt_path
        super().__init__(args, seqs_folder, dataset2transform)


def make_transforms_for_mot17(image_set, args=None):

    normalize = T.MotCompose([
        T.MotToTensor(),
        T.MotNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    scales = [608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992]

    if image_set == 'train':
        return T.MotCompose([
            # T.MotRandomHorizontalFlip(),
            # T.MotRandomSelect(
            #     T.MotRandomResize(scales, max_size=1536),
            #     T.MotCompose([
            #         T.MotRandomResize([800, 1000, 1200]),
            #         T.FixedMotRandomCrop(800, 1200),
            #         T.MotRandomResize(scales, max_size=1536),
            #     ])
            # ),
            T.MotRandomResize(scales, max_size=1536),
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

    dataset2transform_train = mot17_train
    dataset2transform_val = mot17_test
    if image_set == 'train':
        return dataset2transform_train
    elif image_set == 'val':
        return dataset2transform_val
    else:
        raise NotImplementedError()

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

def build(image_set, args):
    root = Path(args.mot_path)
    assert root.exists(), f'provided MOT path {root} does not exist'
    dataset2transform = build_dataset2transform(args, image_set)
    query_transform = get_query_transforms(image_set)
    if image_set == 'train':
        data_txt_path = args.data_txt_path_train
        dataset = ImageNetPseudoVideoDataset(args, data_txt_path=data_txt_path, seqs_folder=root, dataset2transform=dataset2transform
                                             ,query_transform=query_transform)
    if image_set == 'val':
        data_txt_path = args.data_txt_path_val
        dataset = ImageNetPseudoVideoDataset(args, data_txt_path=data_txt_path, seqs_folder=root, dataset2transform=dataset2transform
                                             ,query_transform=query_transform)
    return dataset
