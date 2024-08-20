from typing import Sequence, Tuple, Union

import torch.nn as nn
from torch import nn, cat, add
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.nets.vit import ViT
from monai.utils import ensure_tuple_rep
import torch.nn.functional as F
import torch

class conv_bias(nn.Module):
    def __init__(self, in_ch, out_ch, bias_size=1):
        super(conv_bias, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.merge = nn.Conv3d(out_ch, bias_size, 1)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        x_bias = self.merge(x)
        return x_bias, x


class MetaFDTR(nn.Module):
    def __init__(self, in_channels, out_channels, img_size=(128, 128, 128),
                 depth=(16, 16, 32, 32, 64, 64, 128, 128, 256, 256, 128, 128, 64, 64, 32, 32, 16, 16), bias=4,
                 feature_size=16, hidden_size=768, mlp_dim=3072, num_heads=12, pos_embed="conv",
                 norm_name="instance", conv_block=True, res_block=True, dropout_rate=0.0, spatial_dims=3):
        super(MetaFDTR, self).__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.num_layers = 12

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.patch_size = ensure_tuple_rep(16, spatial_dims)
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
        self.hidden_size = hidden_size
        self.classification = False
        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            classification=self.classification,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )
        self.encoder1 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size, 
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 2, 
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 4, 
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 16,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )


        self.depth = depth
        self.conv0 = conv_bias(in_channels, depth[0], bias_size=bias)
        self.conv1 = conv_bias(depth[0], depth[1], bias_size=bias)
        
        in_chan = bias
        self.conv2 = conv_bias(depth[1]+ in_chan, depth[2], bias_size=bias)

        in_chan = in_chan + bias
        self.conv3 = conv_bias(depth[2] + in_chan, depth[3], bias_size=bias)

        in_chan = in_chan + bias
        self.conv4 = conv_bias(depth[3] + in_chan, depth[4], bias_size=bias)

        in_chan = in_chan + bias
        self.conv5 = conv_bias(depth[4] + in_chan, depth[5], bias_size=bias)

        in_chan = in_chan + bias
        self.conv6 = conv_bias(depth[5] + in_chan, depth[6], bias_size=bias)

        in_chan = in_chan + bias
        self.conv7 = conv_bias(depth[6] + in_chan, depth[7], bias_size=bias)

        in_chan = in_chan + bias
        self.conv8 = conv_bias(depth[7] + in_chan, depth[8], bias_size=bias)

        in_chan = in_chan + bias
        self.conv9 = conv_bias(depth[8] + in_chan, depth[9], bias_size=bias)
        
        #------------------------------------------------------------------------------------------
        # define weights according to the structural order
        self.vars = nn.ParameterList()

        # Weight_Up
        index_i = 9
        for i in range(9,12):
            
            in_chan = in_chan + bias
            w = nn.Parameter(torch.ones(depth[index_i+1],depth[index_i] + in_chan,  3, 3, 3))
            torch.nn.init.kaiming_normal_(w)
            self.vars.append(w)
            self.vars.append(nn.Parameter(torch.zeros(depth[index_i+1])))
            self.vars.append(nn.Parameter(torch.ones(depth[index_i+1])))
            self.vars.append(nn.Parameter(torch.zeros(depth[index_i+1])))
            w = nn.Parameter(torch.ones(4,depth[index_i+1], 1, 1, 1))
            torch.nn.init.kaiming_normal_(w)
            self.vars.append(w)
            self.vars.append(nn.Parameter(torch.zeros(4)))
            index_i=index_i+1

            in_chan = in_chan + bias
            w = nn.Parameter(torch.ones(depth[index_i+1],depth[index_i] + in_chan,  3, 3, 3))
            torch.nn.init.kaiming_normal_(w)
            self.vars.append(w)
            self.vars.append(nn.Parameter(torch.zeros(depth[index_i+1])))
            self.vars.append(nn.Parameter(torch.ones(depth[index_i+1])))
            self.vars.append(nn.Parameter(torch.zeros(depth[index_i+1])))
            w = nn.Parameter(torch.ones( 4, depth[index_i+1],1, 1, 1))
            torch.nn.init.kaiming_normal_(w)
            self.vars.append(w)
            self.vars.append(nn.Parameter(torch.zeros(4)))
            index_i=index_i+1

        index_i = 15
            
        in_chan = in_chan + bias
        w = nn.Parameter(torch.ones(depth[index_i+1],depth[index_i] + in_chan,  3, 3, 3))
        torch.nn.init.kaiming_normal_(w)
        self.vars.append(w)
        self.vars.append(nn.Parameter(torch.zeros(depth[index_i+1])))
        self.vars.append(nn.Parameter(torch.ones(depth[index_i+1])))
        self.vars.append(nn.Parameter(torch.zeros(depth[index_i+1])))
        

        # Weight_Up
        index_i = 16
            
        in_chan = in_chan + bias
        w = nn.Parameter(torch.ones(depth[index_i+1],depth[index_i] + in_chan,  3, 3, 3))
        torch.nn.init.kaiming_normal_(w)
        self.vars.append(w)
        self.vars.append(nn.Parameter(torch.zeros(depth[index_i+1])))
        self.vars.append(nn.Parameter(torch.ones(depth[index_i+1])))
        self.vars.append(nn.Parameter(torch.zeros(depth[index_i+1])))


        w_out = nn.Parameter(torch.ones(out_channels,depth[-1],1,1,1))
        torch.nn.init.kaiming_normal_(w_out)
        self.vars.append(w_out)
        self.vars.append(nn.Parameter(torch.zeros(out_channels)))

        #------------------------------------------------------------------------------------------
        self.up_1_0_0 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_1_1_0 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_2_0_1 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_2_0_0 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_2_1_1 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_2_1_0 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_3_0_2 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_3_0_1 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_3_0_0 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_3_1_2 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_3_1_1 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_3_1_0 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_4_0_3 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_4_0_2 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_4_0_1 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_4_0_0 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_4_1_3 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_4_1_2 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_4_1_1 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_4_1_0 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_3_2_2 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_3_2_1 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_3_2_0 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_3_3_2 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_3_3_1 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_3_3_0 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_2_2_1 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_2_2_0 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_2_3_1 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_2_3_0 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_1_2_0 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_1_3_0 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)

        self.down_0_0_1 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.down_0_0_2 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.down_0_0_3 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.down_0_0_4 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)

        self.down_0_1_1 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.down_0_1_2 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.down_0_1_3 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.down_0_1_4 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)

        self.down_1_0_2 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.down_1_0_3 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.down_1_0_4 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)

        self.down_1_1_2 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.down_1_1_3 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.down_1_1_4 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)

        self.down_2_0_3 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.down_2_0_4 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)

        self.down_2_1_3 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.down_2_1_4 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)

        self.down_3_0_4 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)

        self.down_3_1_4 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)

        self.maxpooling = nn.MaxPool3d(2)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)


        self.proj_axes = (0, spatial_dims + 1) + tuple(d + 1 for d in range(spatial_dims))
        self.proj_view_shape = list(self.feat_size) + [self.hidden_size]

    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward(self, x_in, vars=None):
        
        if vars is None:
            vars = self.vars

        Z = x_in.size()[2]
        Y = x_in.size()[3]
        X = x_in.size()[4]
        diffZ = (16 - x_in.size()[2] % 16) % 16
        diffY = (16 - x_in.size()[3] % 16) % 16
        diffX = (16 - x_in.size()[4] % 16) % 16

        x_in = F.pad(x_in, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2,
                      diffZ // 2, diffZ - diffZ // 2])

        xall, hidden_states_out = self.vit(x_in)

        #block0
        x_bias_0_0_0, x = self.conv0(x_in)
        x_bias_0_1_0, x = self.conv1(x)

        renc0 = x

        x_bias_0_0_1 = self.down_0_0_1(x_bias_0_0_0)
        x_bias_0_0_2 = self.down_0_0_2(x_bias_0_0_1)
        x_bias_0_0_3 = self.down_0_0_3(x_bias_0_0_2)
        x_bias_0_0_4 = self.down_0_0_4(x_bias_0_0_3)

        x_bias_0_1_1 = self.down_0_1_1(x_bias_0_1_0)
        x_bias_0_1_2 = self.down_0_1_2(x_bias_0_1_1)
        x_bias_0_1_3 = self.down_0_1_3(x_bias_0_1_2)
        x_bias_0_1_4 = self.down_0_1_4(x_bias_0_1_3)

        #block1
        x1 = hidden_states_out[0]
        enc1 = self.encoder1(self.proj_feat(x1))

        x_bias_1_0_1, x = self.conv2(cat([enc1, x_bias_0_0_1], dim=1))
        x_bias_1_1_1, x = self.conv3(cat([x, x_bias_0_0_1, x_bias_0_1_1], dim=1))

        renc1 = x
        x_bias_1_0_0 = self.up_1_0_0(x_bias_1_0_1)
        x_bias_1_0_2 = self.down_1_0_2(x_bias_1_0_1)
        x_bias_1_0_3 = self.down_1_0_3(x_bias_1_0_2)
        x_bias_1_0_4 = self.down_1_0_4(x_bias_1_0_3)

        x_bias_1_1_0 = self.up_1_1_0(x_bias_1_1_1)
        x_bias_1_1_2 = self.down_1_1_2(x_bias_1_1_1)
        x_bias_1_1_3 = self.down_1_1_3(x_bias_1_1_2)
        x_bias_1_1_4 = self.down_1_1_4(x_bias_1_1_3)

        #block2
        x2 = hidden_states_out[3]
        enc2 = self.encoder2(self.proj_feat(x2))

        x_bias_2_0_2, x = self.conv4(cat([enc2, x_bias_0_0_2, x_bias_0_1_2, x_bias_1_0_2], dim=1))
        x_bias_2_1_2, x = self.conv5(cat([x, x_bias_0_0_2, x_bias_0_1_2, x_bias_1_0_2, x_bias_1_1_2], dim=1))

        renc2 = x
        x_bias_2_0_1 = self.up_2_0_1(x_bias_2_0_2)
        x_bias_2_0_0 = self.up_2_0_0(x_bias_2_0_1)
        x_bias_2_0_3 = self.down_2_0_3(x_bias_2_0_2)
        x_bias_2_0_4 = self.down_2_0_4(x_bias_2_0_3)

        x_bias_2_1_1 = self.up_2_1_1(x_bias_2_1_2)
        x_bias_2_1_0 = self.up_2_1_0(x_bias_2_1_1)
        x_bias_2_1_3 = self.down_2_1_3(x_bias_2_1_2)
        x_bias_2_1_4 = self.down_2_1_4(x_bias_2_1_3)

        #block3
        x3 = hidden_states_out[6]
        enc3 = self.encoder3(self.proj_feat(x3))

        x_bias_3_0_3, x = self.conv6(
            cat([enc3, x_bias_0_0_3, x_bias_0_1_3, x_bias_1_0_3, x_bias_1_1_3, x_bias_2_0_3], dim=1))
        x_bias_3_1_3, x = self.conv7(cat([x, x_bias_0_0_3, x_bias_0_1_3, x_bias_1_0_3, x_bias_1_1_3, x_bias_2_0_3,
                                          x_bias_2_1_3], dim=1))

        renc3 = x

        x_bias_3_0_2 = self.up_3_0_2(x_bias_3_0_3)
        x_bias_3_0_1 = self.up_3_0_1(x_bias_3_0_2)
        x_bias_3_0_0 = self.up_3_0_0(x_bias_3_0_1)
        x_bias_3_0_4 = self.down_3_0_4(x_bias_3_0_3)

        x_bias_3_1_2 = self.up_3_1_2(x_bias_3_1_3)
        x_bias_3_1_1 = self.up_3_1_1(x_bias_3_1_2)
        x_bias_3_1_0 = self.up_3_1_0(x_bias_3_1_1)
        x_bias_3_1_4 = self.down_3_1_4(x_bias_3_1_3)


        #block4
        x4 = hidden_states_out[9]
        enc4 = self.encoder4(self.proj_feat(x4))

        x_bias_4_0_4, x = self.conv8(
            cat([enc4, x_bias_0_0_4, x_bias_0_1_4, x_bias_1_0_4, x_bias_1_1_4, x_bias_2_0_4, x_bias_2_1_4, x_bias_3_0_4],
                dim=1))
        x_bias_4_1_4, x = self.conv9(cat([x, x_bias_0_0_4, x_bias_0_1_4, x_bias_1_0_4, x_bias_1_1_4, x_bias_2_0_4,
                                          x_bias_2_1_4, x_bias_3_0_4, x_bias_3_1_4], dim=1))

        renc4 = x

        x_bias_4_0_3 = self.up_4_0_3(x_bias_4_0_4)
        x_bias_4_0_2 = self.up_4_0_2(x_bias_4_0_3)
        x_bias_4_0_1 = self.up_4_0_1(x_bias_4_0_2)
        x_bias_4_0_0 = self.up_4_0_0(x_bias_4_0_1)

        x_bias_4_1_3 = self.up_4_1_3(x_bias_4_1_4)
        x_bias_4_1_2 = self.up_4_1_2(x_bias_4_1_3)
        x_bias_4_1_1 = self.up_4_1_1(x_bias_4_1_2)
        x_bias_4_1_0 = self.up_4_1_0(x_bias_4_1_1)

        # block5
        x = self.up(self.encoder5(self.proj_feat(xall)))

        wi = 0

        # -------------------------------------------------------------------------
        up_output = F.conv3d(cat([x, x_bias_0_0_3, x_bias_0_1_3, x_bias_1_0_3, x_bias_1_1_3, x_bias_2_0_3, x_bias_2_1_3, x_bias_3_0_3,
                 x_bias_3_1_3, x_bias_4_0_3], dim=1),
                   vars[wi], vars[wi + 1], padding=1)
        wi += 2
        gn = F.group_norm(up_output, num_groups=up_output.shape[1] // 4, 
                                    weight=vars[wi], bias=vars[wi + 1]
                                    )
        meta_x = F.relu(gn)
        wi += 2

        meta_x_bias = F.conv3d(meta_x, vars[wi], vars[wi + 1], padding=0)
        wi += 2
        x_bias_3_2_3 = meta_x_bias
        x = meta_x

        up_output = F.conv3d(cat([x, x_bias_0_0_3, x_bias_0_1_3, x_bias_1_0_3, x_bias_1_1_3, x_bias_2_0_3, x_bias_2_1_3, x_bias_3_0_3,
                 x_bias_3_1_3, x_bias_4_0_3, x_bias_4_1_3], dim=1),
                   vars[wi], vars[wi + 1], padding=1)
        wi += 2
        gn = F.group_norm(up_output, num_groups=up_output.shape[1] // 4, 
                                    weight=vars[wi], bias=vars[wi + 1]
                                    )

        meta_x = F.relu(gn)
        wi += 2

        meta_x_bias = F.conv3d(meta_x, vars[wi], vars[wi + 1], padding=0)
        wi += 2
        x_bias_3_3_3 = meta_x_bias

        x = meta_x

        rdnc4 = x
        x_bias_3_2_2 = self.up_3_2_2(x_bias_3_2_3)
        x_bias_3_2_1 = self.up_3_2_1(x_bias_3_2_2)
        x_bias_3_2_0 = self.up_3_2_0(x_bias_3_2_1)

        x_bias_3_3_2 = self.up_3_3_2(x_bias_3_3_3)
        x_bias_3_3_1 = self.up_3_3_1(x_bias_3_3_2)
        x_bias_3_3_0 = self.up_3_3_0(x_bias_3_3_1)

        # block6
        x = self.up(x)

        up_output = F.conv3d(cat([x, x_bias_0_0_2, x_bias_0_1_2, x_bias_1_0_2, x_bias_1_1_2, x_bias_2_0_2,
                                          x_bias_2_1_2, x_bias_3_0_2, x_bias_3_1_2, x_bias_4_0_2, x_bias_4_1_2,
                                          x_bias_3_2_2], dim=1),
                   vars[wi], vars[wi + 1], padding=1)
        wi += 2
        gn = F.group_norm(up_output, num_groups=up_output.shape[1] // 4, 
                                    weight=vars[wi], bias=vars[wi + 1]
                                    )
        meta_x = F.relu(gn)
        wi += 2

        meta_x_bias = F.conv3d(meta_x, vars[wi], vars[wi + 1], padding=0)
        wi += 2
        x_bias_2_2_2 = meta_x_bias
        x = meta_x

        up_output = F.conv3d(cat([x, x_bias_0_0_2, x_bias_0_1_2, x_bias_1_0_2, x_bias_1_1_2, x_bias_2_0_2,
                                          x_bias_2_1_2, x_bias_3_0_2, x_bias_3_1_2, x_bias_4_0_2, x_bias_4_1_2,
                                          x_bias_3_2_2, x_bias_3_3_2], dim=1),
                   vars[wi], vars[wi + 1], padding=1)
        wi += 2
        gn = F.group_norm(up_output, num_groups=up_output.shape[1] // 4, 
                                    weight=vars[wi], bias=vars[wi + 1]
                                    )
        meta_x = F.relu(gn)
        wi += 2


        meta_x_bias = F.conv3d(meta_x, vars[wi], vars[wi + 1], padding=0)
        wi += 2
        x_bias_2_3_2 = meta_x_bias
        x = meta_x

        rdnc3 = x
        x_bias_2_2_1 = self.up_2_2_1(x_bias_2_2_2)
        x_bias_2_2_0 = self.up_2_2_0(x_bias_2_2_1)

        x_bias_2_3_1 = self.up_2_3_1(x_bias_2_3_2)
        x_bias_2_3_0 = self.up_2_3_0(x_bias_2_3_1)

        # block7
        x = self.up(x)


        up_output = F.conv3d(cat([x, x_bias_0_0_1, x_bias_0_1_1, x_bias_1_0_1, x_bias_1_1_1, x_bias_2_0_1,
                                           x_bias_2_1_1, x_bias_3_0_1, x_bias_3_1_1, x_bias_4_0_1, x_bias_4_1_1,
                                           x_bias_3_2_1, x_bias_3_3_1, x_bias_2_2_1], dim=1),
                   vars[wi], vars[wi + 1], padding=1)
        wi += 2
        gn = F.group_norm(up_output, num_groups=up_output.shape[1] // 4, 
                                    weight=vars[wi], bias=vars[wi + 1]
                                    )
        meta_x = F.relu(gn)
        wi += 2


        meta_x_bias = F.conv3d(meta_x, vars[wi], vars[wi + 1], padding=0)
        wi += 2
        x_bias_1_2_1 = meta_x_bias
        x = meta_x

        up_output = F.conv3d(cat([x, x_bias_0_0_1, x_bias_0_1_1, x_bias_1_0_1, x_bias_1_1_1, x_bias_2_0_1,
                                           x_bias_2_1_1, x_bias_3_0_1, x_bias_3_1_1, x_bias_4_0_1, x_bias_4_1_1,
                                           x_bias_3_2_1, x_bias_3_3_1, x_bias_2_2_1, x_bias_2_3_1], dim=1),
                   vars[wi], vars[wi + 1], padding=1)
        wi += 2
        gn = F.group_norm(up_output, num_groups=up_output.shape[1] // 4, 
                                    weight=vars[wi], bias=vars[wi + 1]
                                    )
        meta_x = F.relu(gn)
        wi += 2


        meta_x_bias = F.conv3d(meta_x, vars[wi], vars[wi + 1], padding=0)
        wi += 2
        x_bias_1_3_1 = meta_x_bias
        x = meta_x

        rdnc2 = x
        x_bias_1_2_0 = self.up_1_2_0(x_bias_1_2_1)
        x_bias_1_3_0 = self.up_1_3_0(x_bias_1_3_1)

        # block8
        x = self.up(x)

        up_output = F.conv3d(cat([x, x_bias_0_0_0, x_bias_0_1_0, x_bias_1_0_0, x_bias_1_1_0, x_bias_2_0_0,
                                           x_bias_2_1_0, x_bias_3_0_0, x_bias_3_1_0, x_bias_4_0_0, x_bias_4_1_0,
                                           x_bias_3_2_0, x_bias_3_3_0, x_bias_2_2_0, x_bias_2_3_0, x_bias_1_2_0], dim=1),
                   vars[wi], vars[wi + 1], padding=1)
        wi += 2
        gn = F.group_norm(up_output, num_groups=up_output.shape[1] // 4, 
                                    weight=vars[wi], bias=vars[wi + 1]
                                    )
        meta_x = F.relu(gn)
        wi += 2


        
        x = meta_x

        up_output = F.conv3d(cat([x, x_bias_0_0_0, x_bias_0_1_0, x_bias_1_0_0, x_bias_1_1_0, x_bias_2_0_0,
                                           x_bias_2_1_0, x_bias_3_0_0, x_bias_3_1_0, x_bias_4_0_0, x_bias_4_1_0,
                                           x_bias_3_2_0, x_bias_3_3_0, x_bias_2_2_0, x_bias_2_3_0, x_bias_1_2_0,
                                           x_bias_1_3_0], dim=1),
                   vars[wi], vars[wi + 1], padding=1)
        wi += 2
        gn = F.group_norm(up_output, num_groups=up_output.shape[1] // 4, 
                                    weight=vars[wi], bias=vars[wi + 1]
                                    )
        meta_x = F.relu(gn)
        wi += 2

        
        x = meta_x

        rdnc1 =x 

        x = F.conv3d(x, vars[wi], vars[wi+1])

        # Final_activate
        x = F.softmax(x, dim=1)


        return renc0,rdnc1,renc1,rdnc2,renc2,rdnc3,renc3,rdnc4,x[:, :, diffZ//2: Z+diffZ//2, diffY//2: Y+diffY//2, diffX // 2:X + diffX // 2]

    # redefine parameters For Meta
    def part_parameters(self):
        return self.vars