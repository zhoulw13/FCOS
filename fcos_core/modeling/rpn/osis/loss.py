import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

class OSISLossComputation():
    """
        This class computes the OSIS semantic and instance loss.
    """

    def __init__(self, cfg):
        self.semantic_loss_func = nn.BCEWithLogitsLoss()
        self.instance_loss_func = nn.BCELoss()

        self.num_instances = cfg.MODEL.OSIS.NUM_INSTANCES
        self.num_classes = cfg.MODEL.OSIS.NUM_CLASSES - 1

    def prepare_targets(self, targets, feature_size):
        #targets[0].get_field('masks').convert('mask').get_mask_tensor()
        semantics_targets = []
        instances_targets = []
        for idx, size in enumerate(feature_size):
            semantics_targets.append(self.prepare_semantic_target(targets, size, idx))
            instances_targets.append(self.prepare_instance_target(targets, size, idx))

        return semantics_targets, instances_targets

    def prepare_semantic_target(self, targets, feature_size, fpn_level_idx):
        h, w = feature_size
        semantics_targets = torch.zeros((len(targets), self.num_classes, h, w), dtype=torch.uint8)

        for i in range(len(targets)):
            targets_per_im = targets[i]
            masks_per_im = targets_per_im.get_field('masks').convert('mask').get_mask_tensor()
            labels_per_im = targets_per_im.get_field("labels")

            if len(masks_per_im.size()) < 3:
                masks_per_im = masks_per_im.unsqueeze(0)

            with torch.no_grad():
                resized_masks_per_im = F.adaptive_avg_pool2d(Variable(masks_per_im.float()), (h, w)).data

            for j in range(len(labels_per_im)):
                semantics_targets[i, labels_per_im[j]-1] = resized_masks_per_im[j].type(torch.uint8) + (1 - resized_masks_per_im[j].type(torch.uint8)) * semantics_targets[i, labels_per_im[j]-1]

        assert semantics_targets.max() <= 1

        return semantics_targets.permute(0, 2, 3, 1).reshape(-1, self.num_classes).float()

    def prepare_instance_target(self, targets, feature_size, fpn_level_idx):
        h, w = feature_size
        instances_targets = torch.zeros((len(targets), self.num_instances[fpn_level_idx], h, w), dtype=torch.uint8)

        for i in range(len(targets)):
            masks = targets[i].get_field('masks').convert('mask').get_mask_tensor()
            if len(masks.size()) < 3:
                masks = masks.unsqueeze(0)
            with torch.no_grad():
                resized_masks = F.adaptive_avg_pool2d(Variable(masks.float()), feature_size).data

            instance_map = self.compute_single_instancemap(resized_masks, feature_size)

            stride = np.sqrt(h * w / self.num_instances[fpn_level_idx])
            for j in range(self.num_instances[fpn_level_idx]):
                row = int(j * stride // w * stride)
                col = int((j * stride) % w)
                if instance_map[row, col] != 0:
                    instances_targets[i, j] = resized_masks[int(instance_map[row, col]-1)]

        return instances_targets
           

    def compute_single_instancemap(self, masks, feature_size):
        instance_map = torch.zeros(feature_size, dtype=torch.uint8);
        
        for i in range(len(masks)):
            instance_map = (masks[i] > 0.5).type(torch.uint8) * (i+1) + instance_map * (1 - (masks[i] > 0.5).type(torch.uint8))

        assert instance_map.max() < self.num_classes
        return instance_map


    def __call__(self, semantics, instances, targets):
        """
        Arguments:
            semantics (list[Tensor])
            instances (list[Tensor])
            targets (list[BoxList])

        Returns:
        """

        N = len(targets)
        feature_size = []
        for l in range(len(semantics)):
            feature_size.append((semantics[l].size()[-2:]))

        semantics_targets, instances_targets = self.prepare_targets(targets, feature_size)

        semantics_flatten = []
        instances_flatten = []
        for l in range(len(semantics)):
            semantics_flatten.append(semantics[l].permute(0, 2, 3, 1).reshape(-1, self.num_classes))
            instances_flatten.append(instances[l].permute(0, 2, 3, 1).reshape(-1, self.num_instances[l]))

        semantics_flatten = torch.cat(semantics_flatten, dim=0)
        semantics_targets_flatten = torch.cat(semantics_targets, dim=0)

        import pdb
        pdb.set_trace()

        instance_loss = sum([self.instance_loss_func(a, b) for a, b in zip(instances_flatten, instances_targets)])
        instances_flatten = torch.cat(instances_flatten, dim=0)
        instances_targets_flatten = torch.cat(instances_targets, dim=0)

        semantics_loss = self.semantic_loss_func(semantics_flatten, semantics_targets_flatten)
        instances_loss = self.instance_loss_func(instances_flatten, instances_targets_flatten)

        return semantics_loss, instances_loss


def make_osis_loss_evaluator(cfg):
    loss_evaluator = OSISLossComputation(cfg)
    return loss_evaluator