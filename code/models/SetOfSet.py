import torch
from torch import nn
import utils.dataset_utils
from models.baseNet import BaseNet
from models.layers import *
from utils.sparse_utils import SparseMat
from datasets.SceneData import SceneData
from utils import general_utils
from utils.Phases import Phases
# Import the new attention-based layer
from models.attention_layers import AttentionSetOfSetLayer

class SetOfSetBlock(nn.Module):
    def __init__(self, d_in, d_out, conf):
        super(SetOfSetBlock, self).__init__()
        self.block_size = conf.get_int("model.block_size")
        self.use_skip = conf.get_bool("model.use_skip")
        
        # CHANGED: We get the full class path from the config file now.
        type_name = conf.get_string("model.layer_type", default='models.layers.SetOfSetLayer')
        self.layer_type = general_utils.get_class(type_name)
        
        self.layer_kwargs = dict(conf['model'].get('layer_extra_params', {}))

        modules = []
        modules.extend([self.layer_type(d_in, d_out, **self.layer_kwargs), NormalizationLayer()])
        for i in range(1, self.block_size):
            modules.extend([ActivationLayer(), self.layer_type(d_out, d_out, **self.layer_kwargs), NormalizationLayer()])
        self.layers = nn.Sequential(*modules)

        self.final_act = ActivationLayer()

        if self.use_skip:
            if d_in == d_out:
                self.skip = IdentityLayer()
            else:
                self.skip = nn.Sequential(ProjLayer(d_in, d_out), NormalizationLayer())

    def forward(self, x):
        # x is [m,n,d] sparse matrix
        xl = self.layers(x)
        if self.use_skip:
            xl = self.skip(x) + xl

        out = self.final_act(xl)
        return out

class SetOfSetOutliersNet(BaseNet):
    def __init__(self, conf, phase=None):
        super(SetOfSetOutliersNet, self).__init__(conf)
        # n is the number of points and m is the number of cameras
        num_blocks = conf.get_int('model.num_blocks')
        num_feats = conf.get_int('model.num_features')
        multires = conf.get_int('model.multires')

        n_d_out = 3
        m_d_out = self.out_channels
        d_in = 2

        self.embed = EmbeddingLayer(multires, d_in)

        self.equivariant_blocks = torch.nn.ModuleList([SetOfSetBlock(self.embed.d_out, num_feats, conf)])
        for i in range(num_blocks - 1):
            self.equivariant_blocks.append(SetOfSetBlock(num_feats, num_feats, conf))

        self.m_net = get_linear_layers([num_feats] * 2 + [m_d_out], final_layer=True, batchnorm=False)
        self.n_net = get_linear_layers([num_feats] * 2 + [n_d_out], final_layer=True, batchnorm=False)
        self.outlier_net = get_linear_layers([num_feats] * 2 + [1], final_layer=True, batchnorm=False)
        if phase is Phases.FINE_TUNE:
            self.mode = 1
        else:
            self.mode = conf.get_int('train.output_mode', default=3)

        if self.mode == 2:
            for param in self.m_net.parameters():
                param.requires_grad = False
            self.m_net.eval()
            for param in self.n_net.parameters():
                param.requires_grad = False
            self.n_net.eval()

        if self.mode == 1:
            for param in self.outlier_net.parameters():
                param.requires_grad = False
            self.outlier_net.eval()


    def forward(self, data: SceneData):
        x: SparseMat = data.x  # x is [m,n,d] sparse matrix
        x = self.embed(x)
        for eq_block in self.equivariant_blocks:
            x = eq_block(x)  # [m,n,d_in] -> [m,n,d_out]

        if self.mode != 1:
            # outliers predictions
            outliers_out = self.outlier_net(x.values)
            outliers_out = torch.sigmoid(outliers_out)
        else:
            outliers_out = None

        if self.mode != 2:
            # Aggregate edge features to get camera and point features
            cam_feats = torch.zeros(x.shape[0], x.values.shape[1], device=x.values.device).scatter_add_(
                0, x.indices[0, :].unsqueeze(1).expand(-1, x.values.shape[1]), x.values
            )
            point_feats = torch.zeros(x.shape[1], x.values.shape[1], device=x.values.device).scatter_add_(
                0, x.indices[1, :].unsqueeze(1).expand(-1, x.values.shape[1]), x.values
            )
            
            # Cameras predictions
            m_out = self.m_net(cam_feats)
            
            # Points predictions
            n_out = self.n_net(point_feats).T
            
            # predict extrinsic matrix
            pred_cam = self.extract_model_outputs(m_out, n_out, data)
        else:
            pred_cam = None

        return pred_cam, outliers_out

class SetOfSetNet(BaseNet):
    def __init__(self, conf):
        super(SetOfSetNet, self).__init__(conf)
        # n is the number of points and m is the number of cameras
        num_blocks = conf.get_int('model.num_blocks')
        num_feats = conf.get_int('model.num_features')
        multires = conf.get_int('model.multires')

        n_d_out = 3
        m_d_out = self.out_channels
        d_in = 2

        self.embed = EmbeddingLayer(multires, d_in)

        self.equivariant_blocks = torch.nn.ModuleList([SetOfSetBlock(self.embed.d_out, num_feats, conf)])
        for i in range(num_blocks - 1):
            self.equivariant_blocks.append(SetOfSetBlock(num_feats, num_feats, conf))

        self.m_net = get_linear_layers([num_feats] * 2 + [m_d_out], final_layer=True, batchnorm=False)
        self.n_net = get_linear_layers([num_feats] * 2 + [n_d_out], final_layer=True, batchnorm=False)
        
    def forward(self, data: SceneData):
        x = data.x # x is [m,n,d] sparse matrix
        x = self.embed(x)
        for eq_block in self.equivariant_blocks:
            x = eq_block(x) # [m,n,d_in] -> [m,n,d_out]
        
        # After the attention layers, we can get the camera and point features
        # from the sparse matrix by aggregating the edge features.
        # This part of the logic is dependent on how your final layers
        # (m_net and n_net) are expecting their input.
        
        # Aggregate edge features to get camera and point features
        cam_feats = torch.zeros(x.shape[0], x.values.shape[1], device=x.values.device).scatter_add_(
            0, x.indices[0, :].unsqueeze(1).expand(-1, x.values.shape[1]), x.values
        )
        point_feats = torch.zeros(x.shape[1], x.values.shape[1], device=x.values.device).scatter_add_(
            0, x.indices[1, :].unsqueeze(1).expand(-1, x.values.shape[1]), x.values
        )
        
        cam_output = self.m_net(cam_feats)
        point_output = self.n_net(point_feats)
        
        return cam_output, point_output