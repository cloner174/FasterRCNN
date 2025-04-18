# PyTorch Pakages
import torch as t
from torch import nn
from torch.nn import functional as F
import torchvision as tv

# Faster R-CNN Packages
from model.frcnn_bottleneck import FasterRCNNBottleneck
from model.utils.backbone import load_vgg16_extractor, load_vgg16_classifier
from model.rpn.region_proposal_network import RPN
from model.utils.misc import normal_init


# Other Utils
from utils.config import opt
from utils import array_tool as at



class FasterRCNNVGG16(FasterRCNNBottleneck):
    """Faster R-CNN based on VGG-16.
    For descriptions on the interface of this model, please refer to
    :class:`model.faster_rcnn.FasterRCNN`.

    Args:
        n_fg_class (int): The number of classes excluding the background.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.

    """

    def __init__(self,n_fg_class=20):
        # feature extraction (Backbone CNN: VGG16)
        extractor = load_vgg16_extractor(pretrained=True,load_basic=True)
        super(FasterRCNNVGG16, self).__init__(
            n_classes=n_fg_class + 1,   # +1 for background
            extractor = extractor,     # feature extraction
            rpn=RPN(scales=[64, 128, 256, 512],
                            ratios=[0.5, 1, 2],
                            rpn_conv=nn.Conv2d(512, 512, 3, 1, 1),
                            rpn_loc=nn.Conv2d(512, 12 * 4, 1, 1),
                            rpn_score=nn.Conv2d(512, 12 * 2, 1, 1)),
            predictor=load_vgg16_classifier(pretrained=True, load_basic=True),
            loc=nn.Linear(4096, (n_fg_class + 1) * 4),
            score=nn.Linear(4096, n_fg_class + 1),
            spatial_scale=1/16.,
            pooling_size=7,
            roi_sigma=opt.roi_sigma)
        # roi pooling
        self.pooling = tv.ops.RoIPool(self.pooling_size,self.spatial_scale)

        # initialize parameters
        normal_init(self.loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)
    
    # NOTE: inherent from FasterRCNN class
    def feature_extraction_layer(self, x):
        return self.extractor(x)
    
    # NOTE: inherent from FasterRCNN Class
    def roi_pooling_layer(self, feature, roi):
        n = roi.shape[0]
        roi = at.totensor(roi).float()

        # add batch index to roi
        rois = t.cat([t.zeros(n, 1).cuda(), roi],dim=1)
        # change roi order to (batch_index, x1, y1, x2, y2)
        rois = rois[:, [0, 2, 1, 4, 3]].contiguous()
   
        return self.pooling(feature, rois)


    def bbox_regression_and_classification_layer(self, pooled_feature):
        
        # flatten roi pooled feature
        pooled_feature = pooled_feature.view(pooled_feature.shape[0],-1)

        # RCNN Head layer
        fc = self.predictor(pooled_feature)

        # bbox regression & classification
        roi_loc = self.loc(fc)
        roi_score = self.score(fc)

        return roi_loc, roi_score

def normal_init(m, mean, stddev):
    """
    weight initalizer: truncated normal and random normal.
    """
    m.weight.data.normal_(mean, stddev)
    m.bias.data.zero_()
