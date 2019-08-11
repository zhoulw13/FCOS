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
        for size in feature_size:
            semantics_targets.append(self.prepare_semanticmap(targets, size))
            #instances_targets.append(self.prepare_instancemap(targets, size))

        return semantics_targets, instances_targets

    def prepare_instancemap(self, targets, feature_size):
        instances_targets = []
        h, w = feature_size
        for i in range(len(targets)):
            pass


    def prepare_semanticmap(self, targets, feature_size):
        h, w = feature_size
        semantics_targets = torch.zeros((len(targets), self.num_classes, h, w), dtype=torch.uint8)

        for i in range(len(targets)):
            targets_per_im = targets[i]
            masks_per_im = targets_per_im.get_field('masks').convert('mask').get_mask_tensor()
            labels_per_im = targets_per_im.get_field("labels")

            with torch.no_grad():
                resized_masks_per_im = F.adaptive_avg_pool2d(Variable(masks_per_im.float()), (h, w)).data

            for j in range(len(labels_per_im)):
                semantics_targets[i, labels_per_im[j]-1] += resized_masks_per_im[j].type(torch.uint8)

        assert semantics_targets.max() <= 1
        return semantics_targets.permute(0, 2, 3, 1).reshape(-1, self.num_classes)

    def compute_targets_for_locations(self, location, targets):
        pass


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

        import pdb
        pdb.set_trace()

        semantics_flatten = []
        for l in range(len(semantics)):
            semantics_flatten.append(semantics[l].permute(0, 2, 3, 1).reshape(-1, self.num_classes))

        semantics_flatten = torch.cat(semantics_flatten, dim=0)
        semantics_targets_flatten = torch.cat(semantics_targets, dim=0)
        instances_targets_flatten = torch.cat(instances_targets, dim=0)


def make_osis_loss_evaluator(cfg):
    loss_evaluator = OSISLossComputation(cfg)
    return loss_evaluator