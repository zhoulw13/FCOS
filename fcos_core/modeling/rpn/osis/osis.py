import math
import torch
import torch.nn.functional as F
from torch import nn

#from .inference import make_osis_postprocessor
from .loss import make_osis_loss_evaluator

#from maskrcnn_benchmark.layers import Scale

class OSISHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(OSISHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = cfg.MODEL.OSIS.NUM_CLASSES - 1

        semantic_tower = []
        instance_tower = []
        for i in range(cfg.MODEL.OSIS.NUM_CONVS):
            semantic_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            semantic_tower.append(nn.GroupNorm(32, in_channels))
            semantic_tower.append(nn.ReLU())
            instance_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            instance_tower.append(nn.GroupNorm(32, in_channels))
            instance_tower.append(nn.ReLU())

        self.add_module('semantic_tower', nn.Sequential(*semantic_tower))
        self.add_module('instance_tower', nn.Sequential(*instance_tower))
        self.semantic_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1,
            padding=1
        )
        self.instance_pred = nn.ModuleList(
            [nn.Conv2d(in_channels, instance_channels, kernel_size=3, stride=1, padding=1) 
            for instance_channels in cfg.MODEL.OSIS.NUM_INSTANCES])

        # initialization
        for modules in [self.semantic_tower, self.instance_tower,
                        self.semantic_logits]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)  

        for modules in self.instance_pred:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)  

        # initialize the bias for focal loss
        #prior_prob = cfg.MODEL.OSIS.PRIOR_PROB
        #bias_value = -math.log((1 - prior_prob) / prior_prob)
        #torch.nn.init.constant_(self.semantic_logits.bias, bias_value)

        #self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, x):
        semantics = []
        instances = []
        for l, feature in enumerate(x):
            semantic_tower = self.semantic_tower(feature)
            semantics.append(self.semantic_logits(semantic_tower))
            instances.append(self.instance_pred[l](self.instance_tower(feature)))
        return semantics, instances

class OSISModule(torch.nn.Module):
    """
    Module for OSIS computation. Takes feature maps from the backbone and
    OSIS outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels):
        super(OSISModule, self).__init__()

        head = OSISHead(cfg, in_channels)

        #box_selector_test = make_osis_postprocessor(cfg)

        loss_evaluator = make_osis_loss_evaluator(cfg)
        self.head = head
        #self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator
        self.fpn_strides = cfg.MODEL.OSIS.FPN_STRIDES

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        semantics, instances = self.head(features)

        from boxx import show
        show(images.tensors.permute(0,2,3,1)[0].int().cpu().numpy()[:,:,::-1], figsize=(10, 10))
 
        if self.training:
            return self._forward_train(
                semantics, instances, targets 
            )
        else:
            return semantics, instances


    def _forward_train(self, semantics, instances, targets):
        loss_semantics, loss_instances = self.loss_evaluator(semantics, instances, targets)
        losses = {
            "loss_seg": loss_semantics,
            "loss_ins": loss_instances,
        }
        return None, losses

    '''
    def _forward_test(self, locations, box_cls, box_regression, centerness, image_sizes):
        boxes = self.box_selector_test(
            locations, box_cls, box_regression, 
            centerness, image_sizes
        )
        return boxes, {}

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations
    '''


def build_osis(cfg, in_channels):
    return OSISModule(cfg, in_channels)