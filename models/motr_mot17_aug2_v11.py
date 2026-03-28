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
DETR model and criterion classes.
"""
import copy
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import List

from util import box_ops, checkpoint
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate, get_rank,
                       is_dist_avail_and_initialized, inverse_sigmoid)
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from models.structures import Instances, Boxes, pairwise_iou, matched_boxlist_iou

from .backbone import build_backbone
from .matcher import build_matcher
from .deformable_transformer_plus import build_deforamble_transformer
from .qim import build as build_query_interaction_layer
from .memory_bank import build_memory_bank
from .deformable_detr import SetCriterion, MLP
from .segmentation import sigmoid_focal_loss
from torchvision.ops import roi_align
from scipy.optimize import linear_sum_assignment
class ClipMatcher(SetCriterion):
    def __init__(self, num_classes,
                        matcher,
                        weight_dict,
                        losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__(num_classes, matcher, weight_dict, losses)
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_loss = True
        self.losses_dict = {}
        self._current_frame_idx = 0

    def initialize_for_single_clip(self, gt_instances: List[Instances]):
        self.gt_instances = gt_instances
        self.num_samples = 0
        self.sample_device = None
        self._current_frame_idx = 0
        self.losses_dict = {}

    def _step(self):
        self._current_frame_idx += 1

    def calc_loss_for_track_scores(self, track_instances: Instances):
        frame_id = self._current_frame_idx - 1
        gt_instances = self.gt_instances[frame_id]
        outputs = {
            'pred_logits': track_instances.track_scores[None],
        }
        device = track_instances.track_scores.device

        num_tracks = len(track_instances)
        src_idx = torch.arange(num_tracks, dtype=torch.long, device=device)
        tgt_idx = track_instances.matched_gt_idxes  # -1 for FP tracks and disappeared tracks

        track_losses = self.get_loss('labels',
                                     outputs=outputs,
                                     gt_instances=[gt_instances],
                                     indices=[(src_idx, tgt_idx)],
                                     num_boxes=1)
        self.losses_dict.update(
            {'frame_{}_track_{}'.format(frame_id, key): value for key, value in
             track_losses.items()})

    def get_num_boxes(self, num_samples):
        num_boxes = torch.as_tensor(num_samples, dtype=torch.float, device=self.sample_device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        return num_boxes

    def get_loss(self, loss, outputs, gt_instances, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, gt_instances, indices, num_boxes, **kwargs)

    def loss_boxes(self, outputs, gt_instances: List[Instances], indices: List[tuple], num_boxes,unmatched_track_idxes=None):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        # We ignore the regression loss of the track-disappear slots.
        #TODO: Make this filter process more elegant.
        filtered_idx = []
        for src_per_img, tgt_per_img in indices:
            keep = tgt_per_img != -1
            filtered_idx.append((src_per_img[keep], tgt_per_img[keep]))
        indices = filtered_idx
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([gt_per_img.boxes[i] for gt_per_img, (_, i) in zip(gt_instances, indices)], dim=0)

        # for pad target, don't calculate regression loss, judged by whether obj_id=-1
        target_obj_ids = torch.cat([gt_per_img.obj_ids[i] for gt_per_img, (_, i) in zip(gt_instances, indices)], dim=0) # size(16)
        mask = (target_obj_ids != -1)

        loss_bbox = F.l1_loss(src_boxes[mask], target_boxes[mask], reduction='none')
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes[mask]),
            box_ops.box_cxcywh_to_xyxy(target_boxes[mask])))

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses

    def loss_labels(self, outputs, gt_instances: List[Instances], indices, num_boxes, log=False,unmatched_track_idxes=None):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        # The matched gt for disappear track query is set -1.
        labels = []
        for gt_per_img, (_, J) in zip(gt_instances, indices):
            labels_per_img = torch.ones_like(J)
            # set labels of track-appear slots to 0.
            if len(gt_per_img) > 0:
                labels_per_img[J != -1] = gt_per_img.labels[J[J != -1]]
            labels.append(labels_per_img)
        target_classes_o = torch.cat(labels)
        target_classes[idx] = target_classes_o

        if unmatched_track_idxes is not None:
            ignore_mask = torch.zeros_like(target_classes, dtype=torch.bool)
            ignore_mask[0][unmatched_track_idxes] = True  # 要忽略的位置
            target_classes = target_classes[~ignore_mask].unsqueeze(dim=0)
            src_logits = src_logits[~ignore_mask].unsqueeze(dim=0)

        if self.focal_loss:
            gt_labels_target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[:, :, :-1]  # no loss for the last (background) class
            gt_labels_target = gt_labels_target.to(src_logits)
            loss_ce = sigmoid_focal_loss(src_logits.flatten(1),
                                             gt_labels_target.flatten(1),
                                             alpha=0.25,
                                             gamma=2,
                                             num_boxes=num_boxes, mean_in_dim1=False)
            loss_ce = loss_ce.sum()
        else:
            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

        return losses

    def match_use_features_cos(self, track_features, current_features,  all_track_query_idxes, all_gt_idxes):
                # 对每个向量进行 L2 正则化（标准化）
        A_normalized = F.normalize(track_features, p=2, dim=1)
        B_normalized = F.normalize(current_features, p=2, dim=1)
        # import pdb;pdb.set_trace()
        # 计算余弦相似度
        cosine_similarity = torch.mm(A_normalized, B_normalized.t())
        C = -cosine_similarity.detach()
        C = C.to('cpu')

        indices = linear_sum_assignment(C)
        device = track_features.device
        src_idx = indices[0]
        tgt_idx = indices[1]
        # concat src and tgt.
        #import pdb;pdb.set_trace()
        new_matched_indices = torch.stack([all_track_query_idxes[src_idx], all_gt_idxes[tgt_idx]],
                                            dim=1).to(device)


        return new_matched_indices

    def compute_L_intra(self,M, Nt_minus_1, Nt):
        # 第一部分的求和 (0 ≤ i, j < Nt-1)
        L1 = M[:Nt_minus_1, :Nt_minus_1].sum()

        # 第二部分的求和 (Nt-1 ≤ i, j < Nt-1 + Nt)
        start = Nt_minus_1
        end = Nt_minus_1 + Nt
        L2 = M[start:end, start:end].sum()

        # 总损失
        L_intra = L1 + L2
        return L_intra
    
    def compute_L_inter(self, M, margin=0.5):
        """
        计算 L_id^inter 损失
        :param M: (n, n) 处理后的张量
        :param margin: 误差边界 m
        :return: L_inter 标量
        """
        # 1. 找到每一行的最大值索引 j*
        j_star = M.argmax(dim=1)  # (n,)

        # 2. 计算 max_{j', j'≠j*} M_{i,j'}
        M_clone = M.clone()
        M_clone[torch.arange(M.shape[0]), j_star] = -float('inf')  # 排除 j*
        second_max = M_clone.max(dim=1).values  # (n,)

        # 3. 计算 max(second_max + m - M_{i, j*}, 0)
        loss_terms = torch.clamp(second_max + margin - M[torch.arange(M.shape[0]), j_star], min=0)

        # 4. 求和
        L_inter = loss_terms.sum()
        
        return L_inter

    def compute_L_cycle(self, M, Nt_minus_1, Nt):
        """
        计算 |M_{i,j} - M_{j,i}| 的和
        其中 i, j 满足：Nt-1 ≤ i < Nt-1 + Nt, 0 ≤ j < Nt-1
        """
        start_i, end_i = Nt_minus_1, Nt_minus_1 + Nt
        start_j, end_j = 0, Nt_minus_1

        # 选取满足条件的子矩阵
        M_ij = M[start_i:end_i, start_j:end_j]
        M_ji = M[start_j:end_j, start_i:end_i].T  # 交换 i, j 后取转置

        # 计算 |M_ij - M_ji| 并求和
        L_symmetry = torch.abs(M_ij - M_ji).sum()
        
        return L_symmetry

    def get_features_loss(self, new_features, old_features):
        
        loss = {}
        sum_loss = 0
        old_features = F.normalize(old_features, p=2, dim=1)
        new_features = F.normalize(new_features, p=2, dim=1)
        all_features = torch.cat((old_features, new_features),dim=0) 
        cosine_similarity = torch.mm(all_features, all_features.t())

        num_old = old_features.shape[0]
        num_new = new_features.shape[0]

        scale = 2 * torch.log(torch.tensor(num_new + num_old + 1.0, device=cosine_similarity.device))
        cosine_similarity = cosine_similarity.fill_diagonal_(-float('inf')) * scale
        M = torch.nn.functional.softmax(cosine_similarity, dim=1)
        if torch.isnan(M).any() or M.shape[0]==0 or M.shape[1]==0:
            pass
        else:
            loss_intra = self.compute_L_intra(M, num_old, num_new)
            loss_inter = self.compute_L_inter(M, margin=0.5)
            loss_cycle = self.compute_L_cycle(M, num_old, num_new)
            sum_loss += (loss_intra + loss_inter + loss_cycle)
        # import pdb; pdb.set_trace()
        if not isinstance(sum_loss, torch.Tensor):
            sum_loss = torch.tensor(sum_loss,dtype=torch.float32,device=new_features.device)
        loss['loss_features'] = sum_loss
        # import pdb;pdb.set_trace()
        return loss

    def match_for_single_frame(self, outputs: dict, box_features):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        gt_instances_i = self.gt_instances[self._current_frame_idx]  # gt instances of i-th image.
        track_instances: Instances = outputs_without_aux['track_instances']
        pred_logits_i = track_instances.pred_logits  # predicted logits of i-th image.
        pred_boxes_i = track_instances.pred_boxes  # predicted boxes of i-th image.

        obj_idxes = gt_instances_i.obj_ids
        obj_idxes_list = obj_idxes.detach().cpu().numpy().tolist()
        obj_idx_to_gt_idx = {obj_idx: gt_idx for gt_idx, obj_idx in enumerate(obj_idxes_list)}
        outputs_i = {
            'pred_logits': pred_logits_i.unsqueeze(0),
            'pred_boxes': pred_boxes_i.unsqueeze(0),
        }

        # step1. inherit and update the previous tracks.
        num_disappear_track = 0
        for j in range(len(track_instances)):
            obj_id = track_instances.obj_idxes[j].item()
            # set new target idx.
            if obj_id >= 0:
                if obj_id in obj_idx_to_gt_idx:
                    track_instances.matched_gt_idxes[j] = obj_idx_to_gt_idx[obj_id]
                else:
                    num_disappear_track += 1
                    track_instances.matched_gt_idxes[j] = -1  # track-disappear case.
            else:
                track_instances.matched_gt_idxes[j] = -1

        full_track_idxes = torch.arange(len(track_instances), dtype=torch.long).to(pred_logits_i.device)
        matched_track_idxes = (track_instances.obj_idxes >= 0)  # occu 
        prev_matched_indices = torch.stack(
            [full_track_idxes[matched_track_idxes], track_instances.matched_gt_idxes[matched_track_idxes]], dim=1).to(
            pred_logits_i.device)

        # step2. select the unmatched slots.
        # note that the FP tracks whose obj_idxes are -2 will not be selected here.
        unmatched_track_idxes = full_track_idxes[track_instances.obj_idxes == -1]

        # step3. select the untracked gt instances (new tracks).
        tgt_indexes = track_instances.matched_gt_idxes
        tgt_indexes = tgt_indexes[tgt_indexes != -1]

        tgt_state = torch.zeros(len(gt_instances_i)).to(pred_logits_i.device)
        tgt_state[tgt_indexes] = 1
        untracked_tgt_indexes = torch.arange(len(gt_instances_i)).to(pred_logits_i.device)[tgt_state == 0]
        # untracked_tgt_indexes = select_unmatched_indexes(tgt_indexes, len(gt_instances_i))
        untracked_gt_instances = gt_instances_i[untracked_tgt_indexes]

        def match_for_single_decoder_layer(unmatched_outputs, matcher):
            new_track_indices = matcher(unmatched_outputs,
                                             [untracked_gt_instances])  # list[tuple(src_idx, tgt_idx)]

            src_idx = new_track_indices[0][0]
            tgt_idx = new_track_indices[0][1]
            # concat src and tgt.
            new_matched_indices = torch.stack([unmatched_track_idxes[src_idx], untracked_tgt_indexes[tgt_idx]],
                                              dim=1).to(pred_logits_i.device)
            return new_matched_indices

        # step4. do matching between the unmatched slots and GTs.
        unmatched_outputs = {
            'pred_logits': track_instances.pred_logits[unmatched_track_idxes].unsqueeze(0),
            'pred_boxes': track_instances.pred_boxes[unmatched_track_idxes].unsqueeze(0),
        }
        new_matched_indices = match_for_single_decoder_layer(unmatched_outputs, self.matcher)

        # step5. update obj_idxes according to the new matching result.
        track_instances.obj_idxes[new_matched_indices[:, 0]] = gt_instances_i.obj_ids[new_matched_indices[:, 1]].long()
        track_instances.matched_gt_idxes[new_matched_indices[:, 0]] = new_matched_indices[:, 1]
        track_instances.box_features[new_matched_indices[:,0]] = box_features[new_matched_indices[:,1]]
        # step6. calculate iou.
        active_idxes = (track_instances.obj_idxes >= 0) & (track_instances.matched_gt_idxes >= 0)
        active_track_boxes = track_instances.pred_boxes[active_idxes]
        if len(active_track_boxes) > 0:
            gt_boxes = gt_instances_i.boxes[track_instances.matched_gt_idxes[active_idxes]]
            active_track_boxes = box_ops.box_cxcywh_to_xyxy(active_track_boxes)
            gt_boxes = box_ops.box_cxcywh_to_xyxy(gt_boxes)
            track_instances.iou[active_idxes] = matched_boxlist_iou(Boxes(active_track_boxes), Boxes(gt_boxes))

        # step7. merge the unmatched pairs and the matched pairs.
        matched_indices = torch.cat([new_matched_indices, prev_matched_indices], dim=0)

        # step8. calculate losses.
        self.num_samples += len(gt_instances_i) + num_disappear_track
        self.sample_device = pred_logits_i.device
        for loss in self.losses:
            new_track_loss = self.get_loss(loss,
                                           outputs=outputs_i,
                                           gt_instances=[gt_instances_i],
                                           indices=[(matched_indices[:, 0], matched_indices[:, 1])],
                                           num_boxes=1)
            self.losses_dict.update(
                {'frame_{}_{}'.format(self._current_frame_idx, key): value for key, value in new_track_loss.items()})

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                unmatched_outputs_layer = {
                    'pred_logits': aux_outputs['pred_logits'][0, unmatched_track_idxes].unsqueeze(0),
                    'pred_boxes': aux_outputs['pred_boxes'][0, unmatched_track_idxes].unsqueeze(0),
                }
                new_matched_indices_layer = match_for_single_decoder_layer(unmatched_outputs_layer, self.matcher)
                matched_indices_layer = torch.cat([new_matched_indices_layer, prev_matched_indices], dim=0)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    l_dict = self.get_loss(loss,
                                           aux_outputs,
                                           gt_instances=[gt_instances_i],
                                           indices=[(matched_indices_layer[:, 0], matched_indices_layer[:, 1])],
                                           num_boxes=1, )
                    self.losses_dict.update(
                        {'frame_{}_aux{}_{}'.format(self._current_frame_idx, i, key): value for key, value in
                         l_dict.items()})
        self._step()
        return track_instances

    def match_for_single_frame_not_use_id(self, outputs: dict, box_features, last_box_features):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        gt_instances_i = self.gt_instances[self._current_frame_idx]  # gt instances of i-th image.
        track_instances: Instances = outputs_without_aux['track_instances']
        pred_logits_i = track_instances.pred_logits  # predicted logits of i-th image.
        pred_boxes_i = track_instances.pred_boxes  # predicted boxes of i-th image.

        device = pred_logits_i.device
        num_tracks = track_instances.pred_boxes.shape[0]
        num_gts = gt_instances_i.boxes.shape[0]

        outputs_i = {
            'pred_logits': pred_logits_i.unsqueeze(0),
            'pred_boxes': pred_boxes_i.unsqueeze(0),
        }

        # 初始化当前帧匹配状态
        track_instances.matched_gt_idxes[:] = -1
        full_track_idxes = torch.arange(num_tracks, dtype=torch.long, device=device)

        # --------------------------------------------------------------------------------
        # Step 1: 提取需要匹配的活跃 Track Queries 和 活跃 GTs
        # --------------------------------------------------------------------------------
        all_track_query_mask = track_instances.obj_idxes >= 0
        all_track_query_idxes = all_track_query_mask.nonzero().flatten()

        all_gt_mask = gt_instances_i.obj_ids >= 0
        all_gt_idxes = all_gt_mask.nonzero().flatten()

        # --------------------------------------------------------------------------------
        # Step 2: Method B+ (Gated Joint Matching) 匹配已存在的轨迹
        # --------------------------------------------------------------------------------
        if len(all_track_query_idxes) > 0 and len(all_gt_idxes) > 0:
            track_boxes = box_ops.box_cxcywh_to_xyxy(track_instances.pred_boxes[all_track_query_idxes])
            gt_boxes = box_ops.box_cxcywh_to_xyxy(gt_instances_i.boxes[all_gt_idxes])
            
            # 1. 计算空间 Cost
            giou_matrix = generalized_box_iou(track_boxes, gt_boxes)
            cost_spatial = 1.0 - giou_matrix
            
            # 2. 计算表观 Cost
            track_features = track_instances.box_features[all_track_query_idxes]
            current_features = box_features[all_gt_idxes]
            track_features_norm = F.normalize(track_features, p=2, dim=1)
            current_features_norm = F.normalize(current_features, p=2, dim=1)
            cos_sim = torch.mm(track_features_norm, current_features_norm.t())
            cost_app = 1.0 - cos_sim
            
            # 3. 动态融合权重
            weight_app = torch.where(giou_matrix > 0, 0.5, 1.5)
            cost_matrix = cost_spatial + weight_app * cost_app 
            
            # 4. 严格 Gating (消除剧烈运动或遮挡带来的脏匹配)
            invalid_mask = (giou_matrix < -0.1) & (cos_sim < 0.5)
            invalid_mask_hard = cos_sim < 0.2
            cost_matrix[invalid_mask | invalid_mask_hard] = 1e6
            
            # 5. 匈牙利算法求解
            C = cost_matrix.cpu().detach().numpy()
            row_ind, col_ind = linear_sum_assignment(C)
            
            # 6. 过滤无效匹配
            valid_matches = []
            for r, c in zip(row_ind, col_ind):
                if C[r, c] < 1e5:  
                    valid_matches.append([all_track_query_idxes[r], all_gt_idxes[c]])
            
            if len(valid_matches) > 0:
                prev_matched_indices = torch.tensor(valid_matches, dtype=torch.int64, device=device)
            else:
                prev_matched_indices = torch.empty((0, 2), dtype=torch.int64, device=device)
        else:
            prev_matched_indices = torch.empty((0, 2), dtype=torch.int64, device=device)

        # --------------------------------------------------------------------------------
        # Step 3: 找出未匹配上的 Slots (包括 Disappear Track 和 Empty Slots) 以及 新出现的 GT
        # --------------------------------------------------------------------------------
        unmatched_detect_idxes = full_track_idxes[track_instances.obj_idxes == -1] # 一直为空的 slot
        
        # 找出之前是活跃的，但这一帧没匹配上的 (Disappear tracks)
        matched_mask = torch.zeros((num_tracks, num_gts), dtype=torch.bool, device=device)
        if len(prev_matched_indices) > 0:
            matched_mask[prev_matched_indices[:, 0], prev_matched_indices[:, 1]] = True
            track_instances.matched_gt_idxes[prev_matched_indices[:, 0]] = prev_matched_indices[:, 1]
            
        unmatched_track_mask = (matched_mask.sum(dim=-1) == 0) & (track_instances.obj_idxes >= 0) 
        unmatched_track_idxes = unmatched_track_mask.nonzero().flatten() # 消失的目标
        
        # 用于与新 GT 匹配的所有空闲 slots
        unmatched_idxes = torch.cat((unmatched_detect_idxes, unmatched_track_idxes), dim=0)

        # 找出未被分配的 GT (New tracks)
        tgt_indexes = track_instances.matched_gt_idxes
        tgt_indexes = tgt_indexes[tgt_indexes != -1]
        tgt_state = torch.zeros(len(gt_instances_i), device=device)
        tgt_state[tgt_indexes] = 1
        untracked_tgt_indexes = torch.arange(len(gt_instances_i), device=device)[tgt_state == 0]

        # --------------------------------------------------------------------------------
        # Step 4: 新目标的生成 (Empty slots vs New GTs) 使用标准 DETR Matcher
        # --------------------------------------------------------------------------------
        unmatched_outputs = {
            'pred_logits': track_instances.pred_logits[unmatched_idxes].unsqueeze(0),
            'pred_boxes': track_instances.pred_boxes[unmatched_idxes].unsqueeze(0),
        }
        untracked_targets = {
            'labels': gt_instances_i.labels[untracked_tgt_indexes],
            'boxes': gt_instances_i.boxes[untracked_tgt_indexes],
        }
        
        # 使用基础 matcher
        match_res = self.matcher(unmatched_outputs, [untracked_targets])
        # 修改点：适配你的 matcher 返回值 (new_track_indices, min_index, _, _)
        new_track_indices = match_res[0] 
        
        src_idx = new_track_indices[0]
        tgt_idx = new_track_indices[1]
        new_matched_indices = torch.stack([unmatched_idxes[src_idx], untracked_tgt_indexes[tgt_idx]], dim=1).to(device)

        # 获取最终没有任何匹配对象的活跃 Track (彻底无归属)
        matches_mask = torch.isin(unmatched_track_idxes, new_matched_indices[:, 0])
        unmatched_track_idxes_final = unmatched_track_idxes[~matches_mask]

        # --------------------------------------------------------------------------------
        # Step 5: 轨迹生命周期管理 (Lifecycle Management)
        # --------------------------------------------------------------------------------
        matched_indices = torch.cat([new_matched_indices, prev_matched_indices], dim=0)

        # 1. 匹配上的轨迹：丢失时间清零
        if len(matched_indices) > 0:
            track_instances.disappear_time[matched_indices[:, 0]] = 0

        # 2. 未匹配上的轨迹：丢失时间累加
        if len(unmatched_track_idxes_final) > 0:
            track_instances.disappear_time[unmatched_track_idxes_final] += 1

        # 3. 区分容忍期与真正消失 (过滤出应该被 Ignore 的 Slots)
        miss_tolerance = 3  # MOT17 建议设为 3-5 帧
        if len(unmatched_track_idxes_final) > 0:
            mask_ignore = track_instances.disappear_time[unmatched_track_idxes_final] <= miss_tolerance
            true_ignore_idxes = unmatched_track_idxes_final[mask_ignore]
        else:
            true_ignore_idxes = torch.empty((0,), dtype=torch.long, device=device)

        num_disappear_track = len(unmatched_track_idxes_final) - len(true_ignore_idxes) # 真正被作为背景的消失数量

        # --------------------------------------------------------------------------------
        # Step 6: 状态更新与 EMA 特征平滑
        # --------------------------------------------------------------------------------
        if len(matched_indices) > 0:
            track_idx = matched_indices[:, 0]
            gt_idx = matched_indices[:, 1]
            
            # 更新匹配关系和 Obj ID
            track_instances.matched_gt_idxes[track_idx] = gt_idx
            track_instances.obj_idxes[track_idx] = gt_instances_i.obj_ids[gt_idx].long()
            
            # === EMA 特征平滑更新逻辑 ===
            alpha = 0.9  # 历史特征保留率
            current_feat = box_features[gt_idx]
            old_feat = track_instances.box_features[track_idx]
            
            # 如果是刚初始化的 slot (全0)，不使用 EMA，直接赋当前特征
            is_initialized = (old_feat.abs().sum(dim=-1) > 0).unsqueeze(-1)
            ema_feat = alpha * old_feat + (1.0 - alpha) * current_feat
            updated_feat = torch.where(is_initialized, ema_feat, current_feat)
            
            # 融合后重新 L2 归一化
            updated_feat = F.normalize(updated_feat, p=2, dim=1)
            track_instances.box_features[track_idx] = updated_feat.detach()

        # --------------------------------------------------------------------------------
        # Step 7: 计算 IoU 与 Losses
        # --------------------------------------------------------------------------------
        active_idxes = (track_instances.obj_idxes >= 0) & (track_instances.matched_gt_idxes >= 0)
        active_track_boxes = track_instances.pred_boxes[active_idxes]
        if len(active_track_boxes) > 0:
            gt_boxes = gt_instances_i.boxes[track_instances.matched_gt_idxes[active_idxes]]
            active_track_boxes = box_ops.box_cxcywh_to_xyxy(active_track_boxes)
            gt_boxes = box_ops.box_cxcywh_to_xyxy(gt_boxes)
            track_instances.iou[active_idxes] = matched_boxlist_iou(Boxes(active_track_boxes), Boxes(gt_boxes))

        # 汇总 Loss
        self.num_samples += len(gt_instances_i) + num_disappear_track
        self.sample_device = device
        
        for loss in self.losses:
            new_track_loss = self.get_loss(loss,
                                           outputs=outputs_i,
                                           gt_instances=[gt_instances_i],
                                           indices=[(matched_indices[:, 0], matched_indices[:, 1])],
                                           num_boxes=1,
                                           unmatched_track_idxes=true_ignore_idxes) # <- 应用真实的 ignore 列表
            self.losses_dict.update(
                {'frame_{}_{}'.format(self._current_frame_idx, key): value for key, value in new_track_loss.items()})
                
        # 计算特征间的对比学习 Loss
        new_feature_loss = self.get_features_loss(box_features, last_box_features)
        self.losses_dict.update(
                {'frame_{}_{}'.format(self._current_frame_idx, key): value for key, value in new_feature_loss.items()})

        # 计算 Auxiliary Outputs Loss
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                matched_indices_layer = matched_indices
                for loss in self.losses:
                    if loss == 'masks':
                        continue
                    l_dict = self.get_loss(loss,
                                           aux_outputs,
                                           gt_instances=[gt_instances_i],
                                           indices=[(matched_indices_layer[:, 0], matched_indices_layer[:, 1])],
                                           num_boxes=1, 
                                           unmatched_track_idxes=true_ignore_idxes) # <- 辅助层也要同步 ignore
                    self.losses_dict.update(
                        {'frame_{}_aux{}_{}'.format(self._current_frame_idx, i, key): value for key, value in l_dict.items()})
                        
        self._step()
        return track_instances
    
    def match_for_single_frame_use_id(self, outputs: dict, box_features, last_box_features):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        gt_instances_i = self.gt_instances[self._current_frame_idx]  # gt instances of i-th image.
        track_instances: Instances = outputs_without_aux['track_instances']
        pred_logits_i = track_instances.pred_logits  # predicted logits of i-th image.
        pred_boxes_i = track_instances.pred_boxes  # predicted boxes of i-th image.

        obj_idxes = gt_instances_i.obj_ids
        obj_idxes_list = obj_idxes.detach().cpu().numpy().tolist()
        obj_idx_to_gt_idx = {obj_idx: gt_idx for gt_idx, obj_idx in enumerate(obj_idxes_list)}
        outputs_i = {
            'pred_logits': pred_logits_i.unsqueeze(0),
            'pred_boxes': pred_boxes_i.unsqueeze(0),
        }

        # step1. inherit and update the previous tracks.
        num_disappear_track = 0
        for j in range(len(track_instances)):
            obj_id = track_instances.obj_idxes[j].item()
            # set new target idx.
            if obj_id >= 0:
                if obj_id in obj_idx_to_gt_idx:
                    track_instances.matched_gt_idxes[j] = obj_idx_to_gt_idx[obj_id]
                else:
                    num_disappear_track += 1
                    track_instances.matched_gt_idxes[j] = -1  # track-disappear case.
            else:
                track_instances.matched_gt_idxes[j] = -1

        full_track_idxes = torch.arange(len(track_instances), dtype=torch.long).to(pred_logits_i.device)
        matched_track_idxes = (track_instances.obj_idxes >= 0)  # occu 
        prev_matched_indices = torch.stack(
            [full_track_idxes[matched_track_idxes], track_instances.matched_gt_idxes[matched_track_idxes]], dim=1).to(
            pred_logits_i.device)

        # step2. select the unmatched slots.
        # note that the FP tracks whose obj_idxes are -2 will not be selected here.
        unmatched_track_idxes = full_track_idxes[track_instances.obj_idxes == -1]

        # step3. select the untracked gt instances (new tracks).
        tgt_indexes = track_instances.matched_gt_idxes
        tgt_indexes = tgt_indexes[tgt_indexes != -1]

        tgt_state = torch.zeros(len(gt_instances_i)).to(pred_logits_i.device)
        tgt_state[tgt_indexes] = 1
        untracked_tgt_indexes = torch.arange(len(gt_instances_i)).to(pred_logits_i.device)[tgt_state == 0]
        # untracked_tgt_indexes = select_unmatched_indexes(tgt_indexes, len(gt_instances_i))
        untracked_gt_instances = gt_instances_i[untracked_tgt_indexes]

        def match_for_single_decoder_layer(unmatched_outputs, matcher):
            new_track_indices = matcher(unmatched_outputs,
                                             [untracked_gt_instances])  # list[tuple(src_idx, tgt_idx)]

            src_idx = new_track_indices[0][0]
            tgt_idx = new_track_indices[0][1]
            # concat src and tgt.
            new_matched_indices = torch.stack([unmatched_track_idxes[src_idx], untracked_tgt_indexes[tgt_idx]],
                                              dim=1).to(pred_logits_i.device)
            return new_matched_indices

        # step4. do matching between the unmatched slots and GTs.
        unmatched_outputs = {
            'pred_logits': track_instances.pred_logits[unmatched_track_idxes].unsqueeze(0),
            'pred_boxes': track_instances.pred_boxes[unmatched_track_idxes].unsqueeze(0),
        }
        new_matched_indices = match_for_single_decoder_layer(unmatched_outputs, self.matcher)

        # step5. update obj_idxes according to the new matching result.
        track_instances.obj_idxes[new_matched_indices[:, 0]] = gt_instances_i.obj_ids[new_matched_indices[:, 1]].long()
        track_instances.matched_gt_idxes[new_matched_indices[:, 0]] = new_matched_indices[:, 1]
        track_instances.box_features[new_matched_indices[:,0]] = box_features[new_matched_indices[:,1]]
        # step6. calculate iou.
        active_idxes = (track_instances.obj_idxes >= 0) & (track_instances.matched_gt_idxes >= 0)
        active_track_boxes = track_instances.pred_boxes[active_idxes]
        if len(active_track_boxes) > 0:
            gt_boxes = gt_instances_i.boxes[track_instances.matched_gt_idxes[active_idxes]]
            active_track_boxes = box_ops.box_cxcywh_to_xyxy(active_track_boxes)
            gt_boxes = box_ops.box_cxcywh_to_xyxy(gt_boxes)
            track_instances.iou[active_idxes] = matched_boxlist_iou(Boxes(active_track_boxes), Boxes(gt_boxes))

        # step7. merge the unmatched pairs and the matched pairs.
        matched_indices = torch.cat([new_matched_indices, prev_matched_indices], dim=0)

        # step8. calculate losses.
        self.num_samples += len(gt_instances_i) + num_disappear_track
        self.sample_device = pred_logits_i.device
        for loss in self.losses:
            new_track_loss = self.get_loss(loss,
                                           outputs=outputs_i,
                                           gt_instances=[gt_instances_i],
                                           indices=[(matched_indices[:, 0], matched_indices[:, 1])],
                                           num_boxes=1)
            self.losses_dict.update(
                {'frame_{}_{}'.format(self._current_frame_idx, key): value for key, value in new_track_loss.items()})
        new_feature_loss = self.get_features_loss(box_features,last_box_features)
        self.losses_dict.update(
                {'frame_{}_{}'.format(self._current_frame_idx, key): value for key, value in new_feature_loss.items()})
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                unmatched_outputs_layer = {
                    'pred_logits': aux_outputs['pred_logits'][0, unmatched_track_idxes].unsqueeze(0),
                    'pred_boxes': aux_outputs['pred_boxes'][0, unmatched_track_idxes].unsqueeze(0),
                }
                new_matched_indices_layer = match_for_single_decoder_layer(unmatched_outputs_layer, self.matcher)
                matched_indices_layer = torch.cat([new_matched_indices_layer, prev_matched_indices], dim=0)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    l_dict = self.get_loss(loss,
                                           aux_outputs,
                                           gt_instances=[gt_instances_i],
                                           indices=[(matched_indices_layer[:, 0], matched_indices_layer[:, 1])],
                                           num_boxes=1, )
                    self.losses_dict.update(
                        {'frame_{}_aux{}_{}'.format(self._current_frame_idx, i, key): value for key, value in
                         l_dict.items()})
        self._step()
        return track_instances
    

    def forward(self, outputs, input_data: dict):
        # losses of each frame are calculated during the model's forwarding and are outputted by the model as outputs['losses_dict].
        losses = outputs.pop("losses_dict")
        num_samples = self.get_num_boxes(self.num_samples)
        for loss_name, loss in losses.items():
            losses[loss_name] /= num_samples
        return losses


class RuntimeTrackerBase(object):
    def __init__(self, score_thresh=0.7, filter_score_thresh=0.6, miss_tolerance=5):
        self.score_thresh = score_thresh
        self.filter_score_thresh = filter_score_thresh
        self.miss_tolerance = miss_tolerance
        self.max_obj_id = 0

    def clear(self):
        self.max_obj_id = 0

    def update(self, track_instances: Instances):
        track_instances.disappear_time[track_instances.scores >= self.score_thresh] = 0
        for i in range(len(track_instances)):
            if track_instances.obj_idxes[i] == -1 and track_instances.scores[i] >= self.score_thresh:
                # print("track {} has score {}, assign obj_id {}".format(i, track_instances.scores[i], self.max_obj_id))
                track_instances.obj_idxes[i] = self.max_obj_id
                self.max_obj_id += 1
            elif track_instances.obj_idxes[i] >= 0 and track_instances.scores[i] < self.filter_score_thresh:
                track_instances.disappear_time[i] += 1
                if track_instances.disappear_time[i] >= self.miss_tolerance:
                    # Set the obj_id to -1.
                    # Then this track will be removed by TrackEmbeddingLayer.
                    track_instances.obj_idxes[i] = -1


class TrackerPostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, track_instances: Instances, target_size) -> Instances:
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits = track_instances.pred_logits
        out_bbox = track_instances.pred_boxes

        prob = out_logits.sigmoid()
        # prob = out_logits[...,:1].sigmoid()
        scores, labels = prob.max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_size
        scale_fct = torch.Tensor([img_w, img_h, img_w, img_h]).to(boxes)
        boxes = boxes * scale_fct[None, :]

        track_instances.boxes = boxes
        track_instances.scores = scores
        track_instances.labels = labels
        track_instances.remove('pred_logits')
        track_instances.remove('pred_boxes')
        return track_instances


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MOTR(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels, criterion, track_embed,
                 aux_loss=True, with_box_refine=False, two_stage=False, memory_bank=None, use_checkpoint=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.num_queries = num_queries
        self.track_embed = track_embed
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.num_classes = num_classes
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels
        self.use_checkpoint = use_checkpoint
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)
        self.post_process = TrackerPostProcess()
        self.track_base = RuntimeTrackerBase()
        self.criterion = criterion
        self.memory_bank = memory_bank
        self.mem_bank_len = 0 if memory_bank is None else memory_bank.max_his_length

        self.features_before_decoder = None
        self.last_box_features = None
        self.box_features = None

    def _generate_empty_tracks(self):
        track_instances = Instances((1, 1))
        num_queries, dim = self.query_embed.weight.shape  # (300, 512)
        device = self.query_embed.weight.device
        track_instances.ref_pts = self.transformer.reference_points(self.query_embed.weight[:, :dim // 2])
        track_instances.query_pos = self.query_embed.weight
        track_instances.output_embedding = torch.zeros((num_queries, dim >> 1), device=device)
        track_instances.obj_idxes = torch.full((len(track_instances),), -1, dtype=torch.long, device=device)
        track_instances.matched_gt_idxes = torch.full((len(track_instances),), -1, dtype=torch.long, device=device)
        track_instances.disappear_time = torch.zeros((len(track_instances), ), dtype=torch.long, device=device)
        track_instances.iou = torch.zeros((len(track_instances),), dtype=torch.float, device=device)
        track_instances.scores = torch.zeros((len(track_instances),), dtype=torch.float, device=device)
        track_instances.track_scores = torch.zeros((len(track_instances),), dtype=torch.float, device=device)
        track_instances.pred_boxes = torch.zeros((len(track_instances), 4), dtype=torch.float, device=device)
        track_instances.pred_logits = torch.zeros((len(track_instances), self.num_classes), dtype=torch.float, device=device)

        mem_bank_len = self.mem_bank_len
        track_instances.mem_bank = torch.zeros((len(track_instances), mem_bank_len, dim // 2), dtype=torch.float32, device=device)
        track_instances.mem_padding_mask = torch.ones((len(track_instances), mem_bank_len), dtype=torch.bool, device=device)
        track_instances.save_period = torch.zeros((len(track_instances), ), dtype=torch.float32, device=device)
        track_instances.box_features = torch.zeros((len(track_instances), 256),  dtype=torch.float32, device=device)

        return track_instances.to(self.query_embed.weight.device)

    def clear(self):
        self.track_base.clear()

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, }
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def _forward_single_image(self, samples, track_instances: Instances):
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        assert mask is not None

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        # 在 decoder 前保存特征
        if self.training:
            self.features_before_decoder = srcs
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(srcs, masks, pos, track_instances.query_pos, ref_pts=track_instances.ref_pts)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        ref_pts_all = torch.cat([init_reference[None], inter_references[:, :, :, :2]], dim=0)
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'ref_pts': ref_pts_all[5]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        out['hs'] = hs[-1]
        return out
    
    def _post_process_single_image(self, frame_res, track_instances, choice_for_loss, is_last):
        with torch.no_grad():
            if self.training:
                track_scores = frame_res['pred_logits'][0, :].sigmoid().max(dim=-1).values
            else:
                track_scores = frame_res['pred_logits'][0, :, 0].sigmoid()

        track_instances.scores = track_scores
        track_instances.pred_logits = frame_res['pred_logits'][0]
        track_instances.pred_boxes = frame_res['pred_boxes'][0]
        track_instances.output_embedding = frame_res['hs'][0]
        if self.training:
            # the track id will be assigned by the mather.
            frame_res['track_instances'] = track_instances
            if choice_for_loss == 'first_frame':
                track_instances = self.criterion.match_for_single_frame(frame_res, self.box_features)
            elif choice_for_loss == 'other_mot17_frames':
                track_instances = self.criterion.match_for_single_frame_not_use_id(frame_res, self.box_features, self.last_box_features)
            elif choice_for_loss == 'other_crowd_frames':
                track_instances = self.criterion.match_for_single_frame_use_id(frame_res, self.box_features, self.last_box_features)
            else:
                raise RuntimeError("no setting of loss")
        else:
            # import pdb;pdb.set_trace()
            frame_res['detect_instances'] = track_instances
            # each track will be assigned an unique global id by the track base.
            self.track_base.update(track_instances)
        if self.memory_bank is not None:
            track_instances = self.memory_bank(track_instances)
            # track_instances.track_scores = track_instances.track_scores[..., 0]
            # track_instances.scores = track_instances.track_scores.sigmoid()
            if self.training:
                self.criterion.calc_loss_for_track_scores(track_instances)
        tmp = {}
        tmp['init_track_instances'] = self._generate_empty_tracks()
        tmp['track_instances'] = track_instances
        if not is_last:
            out_track_instances = self.track_embed(tmp)
            frame_res['track_instances'] = out_track_instances
        else:
            frame_res['track_instances'] = None
        return frame_res

    def box_cxcywh_to_xyxy(self,x):
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
            (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)

    def extract_box_feature(self, features, gt_instances):
        feature = features[0]
        boxes = self.box_cxcywh_to_xyxy(gt_instances.boxes).squeeze(0)
        width_scale, height_scale = feature.shape[2:4]
        boxes_rescaled = boxes.clone()
        if boxes_rescaled.dim() == 1:
            boxes_rescaled = boxes_rescaled.unsqueeze(0)  # 转为 [1, 4]
        boxes_rescaled[:, [0, 2]] *= width_scale  # x_min, x_max
        boxes_rescaled[:, [1, 3]] *= height_scale # y_min, y_max
        
        # 提取 7x7 的局部特征图
        roi_feature = roi_align(feature, [boxes_rescaled], output_size=(7, 7))
        
        if boxes_rescaled.shape[0] > 0:
            # [修改点]: 使用 GAP (Global Average Pooling) 替代 view Flatten
            # 这样可以在保留语义信息的同时，消除特征对目标在框内绝对位置的敏感度
            roi_feature = roi_feature.mean(dim=[2, 3])  # 维度变为 [N, C], 通常 C=256
        else:
            # 处理空张量的情况，保持维度对齐
            num_channels = feature.shape[1]
            roi_feature = torch.zeros((0, num_channels), dtype=torch.float, device=feature.device)

        return roi_feature

    @torch.no_grad()
    def inference_single_image(self, img, ori_img_size, track_instances=None):
        if not isinstance(img, NestedTensor):
            img = nested_tensor_from_tensor_list(img)
        if track_instances is None:
            track_instances = self._generate_empty_tracks()
        res = self._forward_single_image(img,
                                         track_instances=track_instances)
        res = self._post_process_single_image(res, track_instances, 'first_frame', False)

        track_instances = res['track_instances']
        detect_instances = res['detect_instances']
        
        track_instances = self.post_process(track_instances, ori_img_size)
        detect_instances = self.post_process(detect_instances, ori_img_size)
        ret = {'track_instances': track_instances, 'detect_instances': detect_instances}
        if 'ref_pts' in res:
            ref_pts = res['ref_pts']
            img_h, img_w = ori_img_size
            scale_fct = torch.Tensor([img_w, img_h]).to(ref_pts)
            ref_pts = ref_pts * scale_fct[None]
            ret['ref_pts'] = ref_pts
        return ret

    def forward(self, data: dict):
        if self.training:
            self.criterion.initialize_for_single_clip(data['gt_instances'])
        frames = data['imgs']  # list of Tensor.
        gt_instances = data['gt_instances']
        outputs = {
            'pred_logits': [],
            'pred_boxes': [],
        }

        track_instances = self._generate_empty_tracks()
        keys = list(track_instances._fields.keys())
        track_instances_list = []
        for frame_index, (frame,gt) in enumerate(zip(frames, data['gt_instances'])):
            frame.requires_grad = False
            is_first = frame_index == 0
            is_last = frame_index == len(frames) - 1
            if self.use_checkpoint and frame_index < len(frames) - 2:
                def fn(frame, *args):
                    frame = nested_tensor_from_tensor_list([frame])
                    tmp = Instances((1, 1), **dict(zip(keys, args)))
                    frame_res = self._forward_single_image(frame, tmp)
                    return (
                        frame_res['pred_logits'],
                        frame_res['pred_boxes'],
                        frame_res['ref_pts'],
                        frame_res['hs'],
                        *[aux['pred_logits'] for aux in frame_res['aux_outputs']],
                        *[aux['pred_boxes'] for aux in frame_res['aux_outputs']]
                    )

                args = [frame] + [track_instances.get(k) for k in keys]
                params = tuple((p for p in self.parameters() if p.requires_grad))
                tmp = checkpoint.CheckpointFunction.apply(fn, len(args), *args, *params)
                frame_res = {
                    'pred_logits': tmp[0],
                    'pred_boxes': tmp[1],
                    'ref_pts': tmp[2],
                    'hs': tmp[3],
                    'aux_outputs': [{
                        'pred_logits': tmp[4+i],
                        'pred_boxes': tmp[4+5+i],
                    } for i in range(5)],
                }
            else:
                frame = nested_tensor_from_tensor_list([frame])
                frame_res = self._forward_single_image(frame, track_instances)
            self.box_features = self.extract_box_feature(self.features_before_decoder, gt)

            is_MOT17 = data['dataset'][0] == 1
            if is_first:
                frame_res = self._post_process_single_image(frame_res, track_instances, 'first_frame', is_last)
            else:
                if is_MOT17:
                    frame_res = self._post_process_single_image(frame_res, track_instances, 'other_mot17_frames', is_last)
                else:
                    frame_res = self._post_process_single_image(frame_res, track_instances, 'other_crowd_frames', is_last)

            self.last_box_features = self.box_features.detach()
            track_instances = frame_res['track_instances']
            outputs['pred_logits'].append(frame_res['pred_logits'])
            outputs['pred_boxes'].append(frame_res['pred_boxes'])
            track_instances_list.append(track_instances)
            # import pdb;pdb.set_trace()
        # 可视化 begin
        # from .visualizer import ImageVisualizer
        # visualizer = ImageVisualizer(denormalized=True)
        # visualizer.show(data['imgs'], track_instances_list ,box_format='cxcywh')
        # import pdb;pdb.set_trace()
        # 可视化 end
        if not self.training:
            outputs['track_instances'] = track_instances
        else:
            outputs['losses_dict'] = self.criterion.losses_dict
        return outputs


def build(args):
    dataset_to_num_classes = {
        'coco': 91,
        'coco_panoptic': 250,
        'e2e_mot': 1,
        'e2e_dance': 1,
        'e2e_dance_aug2': 1,
        'e2e_joint': 1,
        'e2e_joint_aug1': 1,
        'e2e_joint_aug2': 1,
        'e2e_static_mot': 1,
    }
    assert args.dataset_file in dataset_to_num_classes
    num_classes = dataset_to_num_classes[args.dataset_file]
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)
    d_model = transformer.d_model
    hidden_dim = args.dim_feedforward
    query_interaction_layer = build_query_interaction_layer(args, args.query_interaction_layer, d_model, hidden_dim, d_model*2)

    img_matcher = build_matcher(args)
    num_frames_per_batch = max(args.sampler_lengths)
    weight_dict = {}
    for i in range(num_frames_per_batch):
        weight_dict.update({"frame_{}_loss_ce".format(i): args.cls_loss_coef,
                            'frame_{}_loss_bbox'.format(i): args.bbox_loss_coef,
                            'frame_{}_loss_giou'.format(i): args.giou_loss_coef,
                            'frame_{}_loss_features'.format(i): args.features_loss_coef,
                            })

    # TODO this is a hack
    if args.aux_loss:
        for i in range(num_frames_per_batch):
            for j in range(args.dec_layers - 1):
                weight_dict.update({"frame_{}_aux{}_loss_ce".format(i, j): args.cls_loss_coef,
                                    'frame_{}_aux{}_loss_bbox'.format(i, j): args.bbox_loss_coef,
                                    'frame_{}_aux{}_loss_giou'.format(i, j): args.giou_loss_coef,
                                    })
    if args.memory_bank_type is not None and len(args.memory_bank_type) > 0:
        memory_bank = build_memory_bank(args, d_model, hidden_dim, d_model * 2)
        for i in range(num_frames_per_batch):
            weight_dict.update({"frame_{}_track_loss_ce".format(i): args.cls_loss_coef})
    else:
        memory_bank = None
    losses = ['labels', 'boxes']
    criterion = ClipMatcher(num_classes, matcher=img_matcher, weight_dict=weight_dict, losses=losses)
    criterion.to(device)
    postprocessors = {}
    model = MOTR(
        backbone,
        transformer,
        track_embed=query_interaction_layer,
        num_feature_levels=args.num_feature_levels,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        criterion=criterion,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        memory_bank=memory_bank,
        use_checkpoint=args.use_checkpoint,
    )
    return model, criterion, postprocessors
#完整版本：第一次匹配之后根据最小值和特征筛选，第二次正常匹配（给匹配上的detect query分配id）。不计算未匹配track query的分类loss。
#joint版本，用id做监督
# 2026 3 25 v11 替换之前交集筛选的策略，以及特征更新机制