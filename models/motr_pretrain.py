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

from models.structures import Instances, Boxes, pairwise_iou, matched_boxlist_iou

from .backbone import build_backbone
from .matcher import build_matcher
from .deformable_transformer_plus import build_deforamble_transformer
from .qim import build as build_query_interaction_layer
from .memory_bank import build_memory_bank
from .deformable_detr import SetCriterion, MLP
from .segmentation import sigmoid_focal_loss

from torchvision.ops import nms
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

    def loss_boxes(self, outputs, gt_instances: List[Instances], indices: List[tuple], num_boxes):
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

    def loss_labels(self, outputs, gt_instances: List[Instances], indices, num_boxes, log=False):
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

    def match_for_single_frame(self, outputs: dict):
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
        # import pdb;pdb.set_trace()
        new_matched_indices = match_for_single_decoder_layer(unmatched_outputs, self.matcher)

        # step5. update obj_idxes according to the new matching result.
        track_instances.obj_idxes[new_matched_indices[:, 0]] = gt_instances_i.obj_ids[new_matched_indices[:, 1]].long()
        track_instances.matched_gt_idxes[new_matched_indices[:, 0]] = new_matched_indices[:, 1]

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
        # import pdb;pdb.set_trace()
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

    def check_box(self, new_tracks_idxes, track_instances, iou_threshold=0.5):
        bboxes =  track_instances.pred_boxes[new_tracks_idxes]
        scores =  track_instances.scores[new_tracks_idxes]

        if bboxes.size(0) <= 1:
            return new_tracks_idxes

        # 转 xywh -> xyxy
        bboxes_xyxy = bboxes.clone()
        bboxes_xyxy[:, 2] = bboxes_xyxy[:, 0] + bboxes_xyxy[:, 2]
        bboxes_xyxy[:, 3] = bboxes_xyxy[:, 1] + bboxes_xyxy[:, 3]

        # NMS
        keep = nms(bboxes_xyxy, scores, iou_threshold)

        # 生成新的 mask
        mask = torch.zeros_like(new_tracks_idxes)
        idx_all = new_tracks_idxes.nonzero(as_tuple=False).squeeze(1)
        mask[idx_all[keep]] = True
        return mask

    def process_new_tracks(self, track_instances: Instances):
        mask1 = track_instances.obj_idxes == -1
        mask2 = track_instances.scores >= self.score_thresh
        new_tracks_idxes = mask1 & mask2
    
        new_tracks_idxes = self.check_box(new_tracks_idxes, track_instances)
        
        if (new_tracks_idxes.shape[0] > 300) and (new_tracks_idxes.sum()): #如果有track query并且有新目标
            # import pdb;pdb.set_trace()
            new_tracks_inidces = new_tracks_idxes.nonzero()
            old_boxes = track_instances.pred_boxes[300:]
            num_old_boxes = old_boxes.shape[0]
            old_boxes_xyxy = old_boxes.clone()
            old_boxes_xyxy[:, 2] = old_boxes_xyxy[:, 0] + old_boxes_xyxy[:, 2]
            old_boxes_xyxy[:, 3] = old_boxes_xyxy[:, 1] + old_boxes_xyxy[:, 3]
            for new_tracks_inidce in new_tracks_inidces:
                new_box = track_instances.pred_boxes[new_tracks_inidce]
                new_box_xyxy = new_box.clone()
                new_box_xyxy[:, 2] = new_box_xyxy[:, 0] + new_box_xyxy[:, 2]
                new_box_xyxy[:, 3] = new_box_xyxy[:, 1] + new_box_xyxy[:, 3]
                new_box_xyxy = new_box_xyxy.expand(num_old_boxes, -1)
                # import pdb;pdb.set_trace()
                iou = matched_boxlist_iou(Boxes(new_box_xyxy), Boxes(old_boxes_xyxy))
                # import pdb;pdb.set_trace()
                if iou.max().item() > 0.5:
                    new_tracks_idxes[new_tracks_inidce] = False
        return new_tracks_idxes
    
    def process_old_tracks(self, track_instances: Instances):
        old_tracks_idxes = track_instances.obj_idxes >= 0
        if old_tracks_idxes.sum() == 0: #第一帧
            return track_instances
        old_tracks_idxes = self.check_box(old_tracks_idxes, track_instances, 0.9)
        if old_tracks_idxes.sum() != (track_instances.obj_idxes >= 0).sum():
            indices = track_instances.obj_idxes >= 0
            indices[old_tracks_idxes] = False
            
            track_instances.obj_idxes[indices] = -1
        return track_instances    

    def update(self, track_instances: Instances):
        # import pdb;pdb.set_trace()
        track_instances.disappear_time[track_instances.scores >= self.score_thresh] = 0
        new_tracks_idxes = self.process_new_tracks(track_instances)
        track_instances = self.process_old_tracks(track_instances)
        for i in range(len(track_instances)):
            if track_instances.obj_idxes[i] == -1 and track_instances.scores[i] >= self.score_thresh and new_tracks_idxes[i]:
            # if track_instances.obj_idxes[i] == -1 and track_instances.scores[i] >= self.score_thresh :
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
                 aux_loss=True, with_box_refine=False, two_stage=False, memory_bank=None, use_checkpoint=False, 
                 feature_recon=True, query_shuffle=False, mask_ratio=0.1, num_patches=10):
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
        
        # pretrain
        # import pdb;pdb.set_trace()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # pooling used for the query patch feature
        # align the patch feature dim to query patch dim.
        self.patch2query = nn.Linear(backbone.num_channels[-1], hidden_dim)
        self.num_patches = num_patches
        self.mask_ratio = mask_ratio
        self.feature_recon = feature_recon
        if self.feature_recon:
            # align the transformer feature to the CNN feature, which is used for the feature reconstruction
            self.feature_align = MLP(hidden_dim, hidden_dim, backbone.num_channels[-1], 2)
        self.query_shuffle = query_shuffle
        assert num_queries % num_patches == 0  # for simplicity
        query_per_patch = num_queries // num_patches
        # the attention mask is fixed during the pre-training
        self.attention_mask = torch.ones(self.num_queries, self.num_queries) * float('-inf')
        for i in range(query_per_patch):
            self.attention_mask[i * query_per_patch:(i + 1) * query_per_patch,
            i * query_per_patch:(i + 1) * query_per_patch] = 0

        self.debug_flag = False

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

    def _forward_single_image(self, samples, track_instances: Instances, patches: Tensor , patches_ids: List = None):
        
        #proceess the patches
        # import pdb;pdb.set_trace()
        batch_num_patches = patches.shape[0]
        
        bs = samples.tensors.size(0)
        
        patch_feature = self.backbone(patches)
        patch_feature_gt = self.avgpool(patch_feature[-1]).flatten(1)

        # align the dim of patch feature with object query with a linear layer
        # pay attention to the difference between "torch.repeat" and "torch.repeat_interleave"
        # it converts the input from "1,2,3,4" to "1,2,3,4,1,2,3,4,1,2,3,4" by torch.repeat
        # "1,2,3,4" to "1,1,1,2,2,2,3,3,3,4,4,4" by torch.repeat_interleave, which is our target.
        query_pos = track_instances.query_pos.clone()
        num_track_queries = query_pos.shape[0] - self.num_queries
        # mask some query patch and add query embedding
        # base_query = self.query_embed.weight[:, self.transformer.d_model:]
        # if mask_query_patch.shape[0] != patch_feature.shape[0]:
        #     self.debug_flag = True
        #     patch_query = patch_feature.squeeze(1)
        #     # return 
        # else:
            
        
        
        idx = torch.randperm(self.num_queries) if self.query_shuffle else torch.arange(self.num_queries)
        
        if self.training:
        # for training, it uses fixed number of query patches.
            if num_track_queries == 0: # first frame
                patch_feature = self.patch2query(patch_feature_gt) \
                .view(bs, batch_num_patches, -1) \
                .repeat_interleave(self.num_queries // self.num_patches, dim=1) \
                .permute(1, 0, 2) \
                .contiguous()

                mask_query_patch = (torch.rand(self.num_queries, bs, 1, device=patches.device) > self.mask_ratio).float()
                # mask some query patch and add query embedding
                # base_query = self.query_embed.weight[:, self.transformer.d_model:]
                # import pdb;pdb.set_trace()
                patch_query = (patch_feature * mask_query_patch).squeeze(1)
                query_pos[:, 256:] = query_pos[:, 256:] + patch_query 
                # import pdb;pdb.set_trace()   
                # query_pos[:, :256] = query_pos[:, :256] + patch_query 
                
            else: #fei first frame
                track_ids = track_instances.obj_idxes[self.num_queries:] #目前track query的id（因为iou不合格或者fn的缘故，会比真实的要少）
                track_patches_ids = []
                track_patch_feature_gt = []
                detect_patch_feature_gt = []
                for i, patches_id in enumerate(patches_ids):
                    if patches_id in list(track_ids):
                        
                        track_patch_feature_gt.append(patch_feature_gt[i])
                        track_patches_ids.append(patches_id)
                    else:
                        detect_patch_feature_gt.append(patch_feature_gt[i])
                
                if len(track_patch_feature_gt):
                    num_track_queries = len(track_patch_feature_gt) #更新一下 去除FP后的数量
                    track_patch_feature_gt = torch.stack(track_patch_feature_gt, dim=0)
                    # import pdb;pdb.set_trace()
                    track_patch_feature = self.patch2query(track_patch_feature_gt) \
                    .view(bs, num_track_queries, -1) \
                    .permute(1, 0, 2) \
                    .contiguous()

                    mask_query_patch = (torch.rand(num_track_queries, bs, 1, device=patches.device) > self.mask_ratio).float()
                    track_patch_query = (track_patch_feature * mask_query_patch).squeeze(1)
                    #调整patch顺序，正确加到track query上面
                    id2patch_idx = {int(id_): i for i, id_ in enumerate(track_patches_ids)}

                    # 根据 track_ids 生成重排索引
                       
                    reorder_indices = torch.tensor(
                        [id2patch_idx[int(id_)] for id_ in track_instances.obj_idxes[self.num_queries:] if id_ >= 0],
                        device=patches.device
                    )
                    # import pdb;pdb.set_trace()
                    # import pdb;pdb.set_trace()   
                    # 重排 patches
                    patch_query_reordered = track_patch_query[reorder_indices]
                    # a = query_pos[self.num_queries:self.num_queries + num_track_queries, 256:]
                    # b = patch_query_reordered
                    # if a.shape[0] != b.shape[0]:
                    #     raise RuntimeError(
                    #     f"[Track–Patch mismatch]\n"
                    #     f"a.shape = {a.shape}\n"
                    #     f"b.shape = {b.shape}\n"
                    #     f"num_track_queries = {num_track_queries}\n"
                    #     f"len(reorder_indices) = {len(reorder_indices)}\n"
                    #     f"obj_idxes = {track_instances.obj_idxes[self.num_queries:]}\n"
                    #     f"patches_ids = {patches_ids}"
                    #     f"track_patches_ids = {track_patches_ids}"
                        
                    # )
                    query_pos[self.num_queries:self.num_queries + num_track_queries, 256:] = \
                    query_pos[self.num_queries:self.num_queries + num_track_queries, 256:] + patch_query_reordered
                if len(detect_patch_feature_gt):
                    detect_patch_feature_gt = torch.stack(detect_patch_feature_gt, dim=0)
                    detect_patch_feature = self.patch2query(detect_patch_feature_gt) \
                    .view(bs, batch_num_patches - num_track_queries, -1) \
                    .repeat_interleave(self.num_queries // self.num_patches, dim=1) \
                    .permute(1, 0, 2) \
                    .contiguous()

                    num_detect_queries = self.num_queries // self.num_patches * (self.num_patches - num_track_queries)
                    mask_query_patch = (torch.rand(num_detect_queries, bs, 1, device=patches.device) > self.mask_ratio).float()
                    # mask some query patch and add query embedding
                    # base_query = self.query_embed.weight[:, self.transformer.d_model:]
                    # import pdb;pdb.set_trace()
                    patch_query = (detect_patch_feature * mask_query_patch).squeeze(1)
                    query_pos[:num_detect_queries, 256:] = query_pos[:num_detect_queries, 256:] + patch_query 
                
                # import pdb;pdb.set_trace()
                
                # query_pos[:, :256] = query_pos[:, :256] + patch_query 
            
            
            
        else:
            patch_feature = self.patch2query(patch_feature_gt) \
                .view(bs, batch_num_patches, -1) \
                .repeat_interleave(self.num_queries // self.num_patches, dim=1) \
                .permute(1, 0, 2) \
                .contiguous() 
            num_queries = batch_num_patches * self.num_queries // self.num_patches
            # for test, it supports x query patches, where x<=self.num_queries.
            # import pdb;pdb.set_trace()  
            query_pos[:num_queries, 256:] = patch_feature.squeeze(1) + query_pos[:num_queries, 256:]
        track_instances.query_pos = query_pos

        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        assert mask is not None
        # import pdb;pdb.set_trace()   
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
        # import pdb;pdb.set_trace()
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
    
    def _post_process_single_image(self, frame_res, track_instances, is_last):
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
            track_instances = self.criterion.match_for_single_frame(frame_res)
        else:
            # each track will be assigned an unique global id by the track base.
            self.track_base.update(track_instances)
        if self.memory_bank is not None:
            track_instances = self.memory_bank(track_instances)
            # track_instances.track_scores = track_instances.track_scores[..., 0]
            # track_instances.scores = track_instances.track_scores.sigmoid()
            if self.training:
                self.criterion.calc_loss_for_track_scores(track_instances)
        # import pdb;pdb.set_trace()
        tmp = {}
        tmp['init_track_instances'] = self._generate_empty_tracks()
        tmp['track_instances'] = track_instances
        if not is_last:
            out_track_instances = self.track_embed(tmp)
            frame_res['track_instances'] = out_track_instances
        else:
            frame_res['track_instances'] = None
        return frame_res

    @torch.no_grad()
    def inference_single_image(self, img, ori_img_size, track_instances=None, patches=None):
        if not isinstance(img, NestedTensor):
            img = nested_tensor_from_tensor_list(img)
        if track_instances is None:
            track_instances = self._generate_empty_tracks()
        # import pdb;pdb.set_trace()
        patches = patches.to(track_instances.scores.device)
        res = self._forward_single_image(img,
                                         track_instances=track_instances, patches=patches)
        res = self._post_process_single_image(res, track_instances, False)

        track_instances = res['track_instances']
        track_instances = self.post_process(track_instances, ori_img_size)
        ret = {'track_instances': track_instances}
        if 'ref_pts' in res:
            ref_pts = res['ref_pts']
            img_h, img_w = ori_img_size
            scale_fct = torch.Tensor([img_w, img_h]).to(ref_pts)
            ref_pts = ref_pts * scale_fct[None]
            ret['ref_pts'] = ref_pts
        return ret

    def forward(self, data: dict): #dict_keys(['imgs', 'gt_instances', 'patches'])
        # 可视化 begin
        # from .visualizer import ImageVisualizer
        # visualizer = ImageVisualizer(denormalized=True)
        # visualizer.show(data['imgs'],data['gt_instances'],box_format='cxcywh')
        
        # 可视化 end
        if self.training:
            self.criterion.initialize_for_single_clip(data['gt_instances'])
        frames = data['imgs']  # list of Tensor.
        outputs = {
            'pred_logits': [],
            'pred_boxes': [],
        }

        track_instances = self._generate_empty_tracks()
        keys = list(track_instances._fields.keys())
        track_list = []
        for frame_index, frame in enumerate(frames):
            # import pdb;pdb.set_trace()
            patches = data['patches'][frame_index]
            patches_ids = data['patches_ids'][frame_index]
            frame.requires_grad = False
            is_last = frame_index == len(frames) - 1
            if self.use_checkpoint and frame_index < len(frames) - 2:
                def fn(frame, *args):
                    frame = nested_tensor_from_tensor_list([frame])
                    tmp = Instances((1, 1), **dict(zip(keys, args)))
                    frame_res = self._forward_single_image(frame, tmp, patches, patches_ids)
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
                frame_res = self._forward_single_image(frame, track_instances, patches, patches_ids)
            frame_res = self._post_process_single_image(frame_res, track_instances, is_last)

            track_instances = frame_res['track_instances']
            track_list.append(track_instances)
            outputs['pred_logits'].append(frame_res['pred_logits'])
            outputs['pred_boxes'].append(frame_res['pred_boxes'])
        # 可视化 begin
        # import pdb;pdb.set_trace()
        # self.debug_flag = True
        # if self.debug_flag:
        #     from .visualizer import ImageVisualizer
        #     visualizer = ImageVisualizer(denormalized=True)
        #     visualizer.show(data['imgs'],data['gt_instances'],box_format='cxcywh')
        #     # visualizer.show(data['imgs'],track_list,box_format='cxcywh')
        #     import pdb;pdb.set_trace()

        # from .visualizer import ImageVisualizer
        # visualizer = ImageVisualizer(denormalized=True)
        # visualizer.show(data['imgs'],data['gt_instances'],box_format='cxcywh')
        # # visualizer.show(data['imgs'],track_list,box_format='cxcywh')
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
        'e2e_imagenet': 2,
        'e2e_dance': 1,
        'e2e_dance_pretrain': 2,
        'e2e_dance_aug1': 1,
        'e2e_joint': 1,
        'e2e_joint_aug1': 1,
        'e2e_static_mot': 1,
    }
    assert args.dataset_file in dataset_to_num_classes
    num_classes = dataset_to_num_classes[args.dataset_file]
    device = torch.device(args.device)
    # import pdb;pdb.set_trace()
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
    num_patches = args.num_pictures_row * args.num_pictures_column
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
        num_patches=num_patches,
        # num_patches=args.num_patches,
        feature_recon=args.feature_recon,
        query_shuffle=args.query_shuffle
    )
    return model, criterion, postprocessors
