import re
import torch.nn as nn
import torch
import torch.nn.functional as F
from ..registry import HEADS
from ..utils import ConvModule
from .bbox_head_semantic_dual import BBoxSemanticHeadDual


@HEADS.register_module
class ConvFCSemanticBBoxHeadDualSeFc(BBoxSemanticHeadDual):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

                                /-> cls convs -> cls fcs -> cls
    shared convs -> shared fcs
                                \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_semantic_convs=0,
                 num_semantic_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 semantic_dims=300,
                 conv_cfg=None,
                 norm_cfg=None,
                 *args,
                 **kwargs):
        super(ConvFCSemanticBBoxHeadDualSeFc, self).__init__(*args, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_semantic_convs +
                num_semantic_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_semantic_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_semantic_convs = num_semantic_convs
        self.num_semantic_fcs = num_semantic_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add semantic specific branch
        self.semantic_convs, self.semantic_fcs, self.semantic_last_dim = \
            self._add_conv_fc_branch(
                self.num_semantic_convs, self.num_semantic_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_semantic_fcs == 0:
                self.semantic_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.FFN_list = nn.ModuleList()
        for i in range(self.num_path):
            FFN = nn.ModuleList()
            for j in range(self.num_shared_fcs):
                in_channel = (self.in_channels * self.roi_feat_area if j == 0 else last_layer_dim)
                layer = nn.Linear(in_channel,last_layer_dim)
                FFN.append(layer)
            self.FFN_list.append(FFN)

        self.relu = nn.ReLU(inplace=True)

        self.T_list = nn.ModuleList()
        self.M_list = nn.ModuleList()
        self.d_T_list = nn.ModuleList()
        self.d_M_list = nn.ModuleList()
        for i in range(self.num_path):
            T = nn.Linear(self.semantic_last_dim, semantic_dims)
            if self.with_decoder:
                d_T = nn.Linear(semantic_dims, self.semantic_last_dim)
            if self.voc is not None:
                M = nn.Linear(self.voc.shape[1], self.vec_list[0].shape[0])  # n*300
                if self.with_decoder:
                    d_M = nn.Linear(self.vec_list[0].shape[0], self.voc.shape[1])  # n*300
            else:
                M = nn.Linear(self.vec_list[0].shape[1], self.vec_list[0].shape[1])
                if self.with_decoder:
                    d_M = nn.Linear(self.vec_list[0].shape[1], self.vec_list[0].shape[1])  # n*300
            self.T_list.append(T) # Ts
            self.M_list.append(M) #Ms
            if self.with_decoder:
                self.d_T_list.append(d_T)
                self.d_M_list.append(d_M)
        
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                           self.num_classes)
            self.fc_reg = nn.Linear(self.reg_last_dim, out_dim_reg)

        
    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def init_weights(self):
        super(ConvFCSemanticBBoxHeadDualSeFc, self).init_weights()
        for module_list in [self.shared_fcs, self.semantic_fcs, self.reg_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, res_feats=None, context_feats=None, return_feats=False, resturn_center_feats=False, bg_vector=None):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x_reg = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x_reg = x.view(x.size(0), -1)
            for fc in self.shared_fcs:
                x_reg = self.relu(fc(x_reg))
        # separate branches
        x_reg = x_reg
        
        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.view(x_reg.size(0), -1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        # classification branch
        semantic_score_list = []
        d_semantic_feature_list = []
        # for each  branch 
        for i in range(self.num_path):
            #FFN compress box features
            x_bbox = x.view(x.size(0), -1)
            for layer in self.FFN_list[i]:
                x_bbox = self.relu(layer(x_bbox))
            
            # use T project to semantic space
            x_semantic = self.T_list[i](x_bbox)

            # normlize to unit vector
            seen_vecs = F.normalize(self.vec_list[i],2,0)
            if self.sync_bg:
                with torch.no_grad():
                    seen_vecs[:, 0] = bg_vector
                    if not self.seen_class:
                        self.vec_unseen[:, 0] = bg_vector
            # compute similarity with extra vocabulary    
            semantic_score = torch.mm(x_semantic, self.voc)

            # use M project to semantic space again
            semantic_score = self.M_list[i](semantic_score)
            if self.with_decoder:
                d_semantic_score = self.d_M_list[i](semantic_score)
                d_semantic_feature = torch.mm(d_semantic_score, self.voc.t())
                d_semantic_feature = self.d_T_list[i](d_semantic_feature)
            #compute similarity with seen class   
            semantic_score = torch.mm(semantic_score, seen_vecs)
            #collect branch result   
            semantic_score_list.append(semantic_score)
            d_semantic_feature_list.append(d_semantic_feature)

        # max function integrate
        semantic_score = torch.max(semantic_score_list[0],semantic_score_list[1])

        # regression branch
        if self.with_reg:
            bbox_pred = self.fc_reg(x_reg)
        if self.with_decoder:
            return semantic_score, bbox_pred, x_bbox, d_semantic_feature_list
        else:
            return semantic_score, bbox_pred


@HEADS.register_module
class TCB(ConvFCSemanticBBoxHeadDualSeFc):

    def __init__(self, num_fcs=2, fc_out_channels=1024, *args, **kwargs):
        assert num_fcs >= 1
        super(TCB, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=num_fcs,
            num_semantic_convs=0,
            num_semantic_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
