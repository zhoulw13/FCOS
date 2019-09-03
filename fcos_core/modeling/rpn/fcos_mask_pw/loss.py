"""
This file contains specific functions for computing losses of FCOS
file
"""

import math
import torch
from torch.nn import functional as F
from torch.autograd import Variable
from torch import nn

from ..utils import concat_box_prediction_layers
from fcos_core.layers import IOULoss
from fcos_core.layers import SigmoidFocalLoss
from fcos_core.layers.misc import interpolate
from fcos_core.modeling.matcher import Matcher
from fcos_core.modeling.utils import cat
from fcos_core.structures.boxlist_ops import boxlist_iou
from fcos_core.structures.boxlist_ops import cat_boxlist


INF = 100000000


class FCOSMaskPWLossComputation(object):
    """
    This class computes the FCOS losses.
    """

    def __init__(self, cfg):
        self.cls_loss_func = SigmoidFocalLoss(
            cfg.MODEL.FCOS.LOSS_GAMMA,
            cfg.MODEL.FCOS.LOSS_ALPHA
        )
        # we make use of IOU Loss for bounding boxes regression,
        # but we found that L1 in log scale can yield a similar performance
        self.box_reg_loss_func = IOULoss()
        self.centerness_loss_func = nn.BCEWithLogitsLoss()
        self.box_mask_loss_func = nn.BCEWithLogitsLoss()

        self.box_mask_pw_channels = cfg.MODEL.FCOS.MASK_PW_CHANNELS

    def prepare_targets(self, points, feature_size, targets):
        object_sizes_of_interest = [
            [-1, 64],
            [64, 128],
            [128, 256],
            [256, 512],
            [512, INF],
        ]
        expanded_object_sizes_of_interest = []
        for l, points_per_level in enumerate(points):
            object_sizes_of_interest_per_level = \
                points_per_level.new_tensor(object_sizes_of_interest[l])
            expanded_object_sizes_of_interest.append(
                object_sizes_of_interest_per_level[None].expand(len(points_per_level), -1)
            )

        expanded_object_sizes_of_interest = torch.cat(expanded_object_sizes_of_interest, dim=0)
        num_points_per_level = [len(points_per_level) for points_per_level in points]
        points_all_level = torch.cat(points, dim=0)
        labels, reg_targets, mask_targets = self.compute_targets_for_locations(
            points_all_level, feature_size, targets, expanded_object_sizes_of_interest
        )

        for i in range(len(labels)):
            labels[i] = torch.split(labels[i], num_points_per_level, dim=0)
            reg_targets[i] = torch.split(reg_targets[i], num_points_per_level, dim=0)

        labels_level_first = []
        reg_targets_level_first = []
        mask_targets_level_first = []
        for level in range(len(points)):
            labels_level_first.append(
                torch.cat([labels_per_im[level] for labels_per_im in labels], dim=0)
            )
            reg_targets_level_first.append(
                torch.cat([reg_targets_per_im[level] for reg_targets_per_im in reg_targets], dim=0)
            )
            mask_targets_level_first.append(
                torch.cat([mask_targets_per_im[level].unsqueeze(0) for mask_targets_per_im in mask_targets], dim=0)
            )

        return labels_level_first, reg_targets_level_first, mask_targets_level_first

    def compute_targets_for_locations(self, locations, feature_size, targets, object_sizes_of_interest):
        labels = []
        reg_targets = []
        mask_targets = []
        xs, ys = locations[:, 0], locations[:, 1]
        device = targets[0].bbox.device

        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            assert targets_per_im.mode == "xyxy"
            bboxes = targets_per_im.bbox
            labels_per_im = targets_per_im.get_field("labels")
            masks_per_im = targets_per_im.get_field('masks').get_mask_tensor().to(device)
            area = targets_per_im.area()

            if len(masks_per_im.size()) < 3:
                masks_per_im = masks_per_im.unsqueeze(0)
            instance_mask, instances = self.compute_single_instance_mask(masks_per_im)

            masks = []
            for size in feature_size:
                with torch.no_grad():
                    resized_masks_per_im = interpolate(instance_mask.unsqueeze(0).float(), size=size, mode='bilinear', align_corners=False)#F.adaptive_avg_pool2d(Variable(instance_mask.float()), size).data
                masks.append(instances[resized_masks_per_im.squeeze().long()].permute(2, 0, 1).float())

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

            is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0

            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]) & \
                (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = 0

            #masks = masks[range(len(locations)), locations_to_gt_inds]
            #masks[locations_to_min_area == INF] = 0
            #masks = (masks.sum(dim=1) > 0).float()

            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)
            mask_targets.append(masks)

        return labels, reg_targets, mask_targets

    def compute_single_instance_mask(self, masks):
        instances = torch.split(masks, [1]*len(masks), dim=0)
        instances = sorted(instances, key=lambda x : x.sum(), reverse=True)
        re = instances[0]
        for i, item in enumerate(instances[1:]):
            re = re * (1 - item) + item * (i+2)
        
        size = int(math.sqrt(self.box_mask_pw_channels))
        obj = []
        #obj.append(torch.zeros((1, self.box_mask_pw_channels), device=instances[0].device, dtype=instances[0].dtype))
        for item in instances:
            loc = torch.nonzero(item)
            xmin, xmax, ymin, ymax = loc[:, 1].min(), loc[:, 1].max()+1, loc[:, 2].min(), loc[:, 2].max()+1
            tmp = interpolate(item[:, xmin:xmax, ymin:ymax].unsqueeze(0).float(), size=(size, size), mode='bilinear', align_corners=False) > 0
            obj.append(tmp.squeeze(0).reshape(1, -1))
        obj.insert(0, torch.zeros_like(obj[0]))

        return re, torch.cat(obj, dim=0)

    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                      (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    def __call__(self, locations, box_cls, box_regression, centerness, box_mask, targets):
        """
        Arguments:
            locations (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            centerness (list[Tensor])
            box_mask (list[Tensor])
            targets (list[BoxList])

        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
        """
        N = box_cls[0].size(0)
        num_classes = box_cls[0].size(1)
        feature_size = []
        for l in range(len(box_regression)):
            feature_size.append((box_regression[l].size()[-2:]))

        labels, reg_targets, mask_targets = self.prepare_targets(locations, feature_size, targets)

        box_cls_flatten = []
        box_regression_flatten = []
        centerness_flatten = []
        box_mask_flatten = []
        labels_flatten = []
        reg_targets_flatten = []
        mask_targets_flatten = []
        for l in range(len(labels)):
            box_cls_flatten.append(box_cls[l].permute(0, 2, 3, 1).reshape(-1, num_classes))
            box_regression_flatten.append(box_regression[l].permute(0, 2, 3, 1).reshape(-1, 4))
            labels_flatten.append(labels[l].reshape(-1))
            reg_targets_flatten.append(reg_targets[l].reshape(-1, 4))
            centerness_flatten.append(centerness[l].reshape(-1))
            box_mask_flatten.append(box_mask[l].reshape(-1))
            mask_targets_flatten.append(mask_targets[l].reshape(-1))

        box_cls_flatten = torch.cat(box_cls_flatten, dim=0)
        box_regression_flatten = torch.cat(box_regression_flatten, dim=0)
        centerness_flatten = torch.cat(centerness_flatten, dim=0)
        box_mask_flatten = torch.cat(box_mask_flatten, dim=0)
        labels_flatten = torch.cat(labels_flatten, dim=0)
        reg_targets_flatten = torch.cat(reg_targets_flatten, dim=0)
        mask_targets_flatten = torch.cat(mask_targets_flatten, dim=0)

        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)
        cls_loss = self.cls_loss_func(
            box_cls_flatten,
            labels_flatten.int()
        ) / (pos_inds.numel() + N)  # add N to avoid dividing by a zero

        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]
        box_mask_flatten = box_mask_flatten[pos_inds]
        mask_targets_flatten = mask_targets_flatten[pos_inds]

        if pos_inds.numel() > 0:
            centerness_targets = self.compute_centerness_targets(reg_targets_flatten)
            reg_loss = self.box_reg_loss_func(
                box_regression_flatten,
                reg_targets_flatten,
                centerness_targets
            )
            centerness_loss = self.centerness_loss_func(
                centerness_flatten,
                centerness_targets
            )
            mask_loss = self.box_mask_loss_func(
                box_mask_flatten, 
                mask_targets_flatten
            )
        else:
            reg_loss = box_regression_flatten.sum()
            centerness_loss = centerness_flatten.sum()
            mask_loss = box_mask_flatten.sum()
        

        return cls_loss, reg_loss, centerness_loss, mask_loss


def make_fcos_mask_pw_loss_evaluator(cfg):
    loss_evaluator = FCOSMaskPWLossComputation(cfg)
    return loss_evaluator
