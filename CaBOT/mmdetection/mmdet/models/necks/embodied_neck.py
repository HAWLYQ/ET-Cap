# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import build_positional_encoding, build_transformer_layer_sequence
from mmcv.cnn import xavier_init
from mmcv.runner import BaseModule
from ..builder import NECKS
from mmcv.cnn import Conv2d
import torch.nn.functional as F
from ..decoding_heads.navi_head import _get_causal_mask
from mmcv.utils.logging import get_logger, logger_initialized, print_log



# encode region-level features for each image
@NECKS.register_module()
class NaviImageTransformerNeck(BaseModule):
    def __init__(self,
                in_channels, 
                output_channels, # not used
                encoder, 
                encode_positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
                init_cfg=None):
        super(NaviImageTransformerNeck, self).__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        
        self.encoder = build_transformer_layer_sequence(encoder)
        self.encode_positional_encoding = build_positional_encoding(
            encode_positional_encoding)
        self.embed_dims = self.encoder.embed_dims
        self.input_proj = Conv2d(
            self.in_channels, self.embed_dims, kernel_size=1)
    
    def init_weights(self):
        # follow the official detr to init transformer parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True
    
    def forward(self, x, patch_mean_pool=True, **kwargs):
        """
        x:  [batch, img_seq, c, h, w]
        """
        bs, img_seq, c, h, w = x.shape
        img_num = bs*img_seq
        x = x.reshape(img_num, c, h, w)
        # input_img_h, input_img_w = img_metas[0][0]['batch_input_shape']
        masks = x.new_zeros((img_num, h, w)).to(torch.bool) # False means not mask
        x = self.input_proj(x) # [img_num, 256, h, w]
        # interpolate masks to have the same spatial shape with x
        """asks = F.interpolate(
            masks.unsqueeze(1), size=x.shape[-2:]).to(torch.bool).squeeze(1)"""
        # position encoding

        # print('NaviImageTransformerNeck masks', masks)
        en_pos_embed = self.encode_positional_encoding(masks)  # [img_num, embed_dim, h, w]
        # print('NaviImageTransformerNeck en pos embed', en_pos_embed)
        
        # use `view` instead of `flatten` for dynamically exporting to ONNX
        x = x.view(img_num, self.embed_dims, -1).permute(2, 0, 1)  # [img_num, c, h, w] -> [h*w, img_num, c]
        en_pos_embed = en_pos_embed.reshape(img_num, self.embed_dims, -1).permute(2, 0, 1) # [h*w, img_num, c]
        masks = masks.reshape(img_num, -1)  #  [img_num, h*w]
        # print('NaviImageTransformerNeck before encoder x:', x)
        # print('NaviImageTransformerNeck before encoder pos emb:', en_pos_embed)
        # print('NaviImageTransformerNeck before encoder masks:', masks)
        en_output = self.encoder(
            query=x,
            key=None,
            value=None,
            query_pos=en_pos_embed,
            query_key_padding_mask=masks) 
        # print('NaviImageTransformerNeck after encoder en_output:', en_output)
        
        # en_output: [h*w, bs*img_seq, dim]

        # calculate a single feature for each image 
        en_output = en_output.reshape(h*w, bs, img_seq, -1) # [h*w, bs, img_seq, dim]
        if patch_mean_pool:
            en_output = torch.mean(en_output, dim=0)  # [bs, img_seq, dim]
        # print('NaviImageTransformerNeck en_output.size:', en_output.size())
        return en_output


# exactly same as NaviImageTransformerNeck
@NECKS.register_module()
class NaviImageTransformerPreNeck(NaviImageTransformerNeck):
    def __init__(self,
                in_channels, 
                output_channels, # not used
                encoder, 
                encode_positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
                init_cfg=None):
        super(NaviImageTransformerPreNeck, self).__init__(
                                                        in_channels,
                                                        output_channels, # not used
                                                        encoder, 
                                                        encode_positional_encoding,
                                                        init_cfg)

# 1. encode region-level features for each image (multi layer)
# 2. encode time-level features for same region (multi layer)
@NECKS.register_module()
class NaviImageRegionTimeTransformerNeck(BaseModule):
    def __init__(self,
                in_channels, 
                output_channels, # not used
                region_encoder, 
                time_encoder,
                region_encode_positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
                time_encode_positional_encoding=dict(
                     type='SeqLearnedPositionalEncoding',
                     num_feats=128,
                     normalize=True),
                init_cfg=None):
        super(NaviImageRegionTimeTransformerNeck, self).__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        
        self.region_encoder = build_transformer_layer_sequence(region_encoder)
        self.region_encode_positional_encoding = build_positional_encoding(
            region_encode_positional_encoding)

        self.time_encoder = build_transformer_layer_sequence(time_encoder)
        self.time_encode_positional_encoding = build_positional_encoding(
            time_encode_positional_encoding)
        
        self.embed_dims = self.region_encoder.embed_dims
        self.input_proj = Conv2d(
            self.in_channels, self.embed_dims, kernel_size=1)
    
    def init_weights(self, skip_region_encoder=False):
        # follow the official detr to init transformer parameters
        # anwen hu: if region encoder is init with detr, skip init_weights for these parameters
        if not skip_region_encoder:
            for m in self.modules():
                if hasattr(m, 'weight') and m.weight.dim() > 1:
                    xavier_init(m, distribution='uniform')
            self._is_init = True
        else:
            # Get the initialized logger, if not exist,
            # create a logger named `mmcv`
            logger_names = list(logger_initialized.keys())
            logger_name = logger_names[0] if logger_names else 'mmcv'
            for c_m in self.children():
                if c_m.__class__.__name__ == 'DetrTransformerEncoder':
                    print_log(f'skip {c_m.__class__.__name__}.init_weights()',logger=logger_name)
                else:
                    for m in c_m.modules():
                        # print(m)
                        if hasattr(m, 'weight') and m.weight.dim() > 1:
                            xavier_init(m, distribution='uniform')
            self._is_init = True

    
    def forward_region_encoding(self, x):
        """
        x: [img_num, c, h, w]
        """
        img_num, c, h, w = x.shape
        # input_img_h, input_img_w = img_metas[0][0]['batch_input_shape']
        masks = x.new_zeros((img_num, h, w)).to(torch.bool) # False means not mask
        # interpolate masks to have the same spatial shape with x
        """asks = F.interpolate(
            masks.unsqueeze(1), size=x.shape[-2:]).to(torch.bool).squeeze(1)"""
        # position encoding
        # print('NaviImageRegionTimeTransformerNeck masks', masks)
        en_pos_embed = self.region_encode_positional_encoding(masks)  # [img_num, embed_dim, h, w]
        # print('NaviImageRegionTimeTransformerNeck en pos embed', en_pos_embed)
        
        # use `view` instead of `flatten` for dynamically exporting to ONNX
        x = x.view(img_num, self.embed_dims, -1).permute(2, 0, 1)  # [img_num, c, h, w] -> [h*w, img_num, c]
        en_pos_embed = en_pos_embed.reshape(img_num, self.embed_dims, -1).permute(2, 0, 1) # [h*w, img_num, c]
        masks = masks.reshape(img_num, -1)  #  [img_num, h*w]
        # print('NaviImageRegionTimeTransformerNeck before encoder x:', x)
        # print('NaviImageRegionTimeTransformerNeck before encoder pos emb:', en_pos_embed)
        # print('NaviImageRegionTimeTransformerNeck before encoder masks:', masks)
        """
        attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
        query_key_padding_mask (Tensor): ByteTensor for `query`, with
            shape [bs, num_queries]. Only used in `self_attn` layer.
            Defaults to None.
        key_padding_mask (Tensor): ByteTensor for `query`, with
            shape [bs, num_keys]. Default: None.
        """
        en_output = self.region_encoder(
            query=x,
            key=None,
            value=None,
            query_pos=en_pos_embed,
            query_key_padding_mask=masks) 
         # en_output: [h*w, img_num, dim]
        return en_output
    
    def forward_time_encoding(self, x):
        """
        x: [h*w, batch, img_seq, dim]
        return [h*w*bs, seq, embed_dim]
        """
        region_num, bs, img_seq, c = x.size()
        masks = x.new_zeros((bs, img_seq)).to(torch.bool) # False means not mask
        # only the shape of masks is used during SeqLearnedPositionalEncoding
        encode_pos_embed = self.time_encode_positional_encoding(masks)  # [bs, embed_dim, seq]
        encode_pos_embed = encode_pos_embed.unsqueeze(0).repeat(region_num, 1, 1, 1) # [h*w, bs, embed_dim, seq]
        encode_pos_embed = encode_pos_embed.reshape(region_num*bs, -1, img_seq).permute(2, 0, 1) # [seq, h*w*bs, embed_dim]
        x = x.reshape(region_num*bs, img_seq, c).permute(1, 0, 2)  # [seq, h*w*bs, c]
        # print(x.type(), decode_pos_embed.type(), seq_masks.type())
        attn_mask = _get_causal_mask(img_seq, x.device) # [seq, seq] 
        outs = self.time_encoder(query=x, key=None, value=None,
                                query_pos=encode_pos_embed, 
                                attn_masks=attn_mask)  # [seq, h*w*bs, embed_dim]
        output = outs.transpose(0, 1) # [h*w*bs, seq, embed_dim]
        return output

    def forward(self, x, patch_mean_pool=True, **kwargs):
        """
        x: [batch, img_seq, c, h, w]
        """
        # TODO:
        batch, img_seq, c, h, w = x.shape
        x = x.reshape(batch*img_seq, c, h, w)
        x = self.input_proj(x) # [img_num, 256, h, w]
        region_x = self.forward_region_encoding(x) # [h*w, batch*img_seq, dim] 
        # [h*w, batch*img_seq, dim] > [h*w, batch, img_seq, dim]
        region_x = region_x.reshape(h*w, batch, img_seq, -1)
        output = self.forward_time_encoding(region_x) # [h*w*bs, img_seq, embed_dim]
        # [img_seq, h*w*batch, dim] > [img_seq, h*w, batch, dim] > [img_seq, h*w, batch, dim]
        output = output.reshape(h*w, batch, img_seq, -1) # [h*w, bs, img_seq, dim]
        # calculate a single feature for each image 
        if patch_mean_pool:
            output = torch.mean(output, dim=0)  # [bs, img_seq, dim]
        # print('NaviImageRegionTimeTransformerNeck output.size:', en_output.size())
        return output



# multi block
# for each block
# 1. encode region-level features for each image (single layer)
# 2. encode time-level features for same region (single layer)
@NECKS.register_module()
class NaviImageRegionTimeBlockTransformerNeck(BaseModule):
    def __init__(self,
                in_channels, 
                output_channels,
                #encoder=None, 
                block_num=2,
                region_encoder=None, 
                time_encoder=None,
                region_encode_positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
                time_encode_positional_encoding=dict(
                     type='SeqLearnedPositionalEncoding',
                     num_feats=128,
                     normalize=True),
                init_cfg=None):
        super(NaviImageRegionTimeBlockTransformerNeck, self).__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.block_num = block_num

        assert region_encoder.num_layers == 1
        assert time_encoder.num_layers == 1

        self.block_region_layers = nn.ModuleList([build_transformer_layer_sequence(region_encoder) for i in range(self.block_num)])
        self.block_time_layers = nn.ModuleList([build_transformer_layer_sequence(time_encoder) for i in range(self.block_num)])

        self.region_encode_positional_encoding = build_positional_encoding(
            region_encode_positional_encoding)        
        self.time_encode_positional_encoding = build_positional_encoding(
            time_encode_positional_encoding)
        
        self.embed_dims = output_channels
        self.input_proj = Conv2d(
            self.in_channels, self.embed_dims, kernel_size=1)
    
    def init_weights(self, skip_region_encoder=False):
        # follow the official detr to init transformer parameters
        # anwen hu: if region encoder is init with detr, skip init_weights for these parameters
        if not skip_region_encoder:
            for m in self.modules():
                if hasattr(m, 'weight') and m.weight.dim() > 1:
                    xavier_init(m, distribution='uniform')
            self._is_init = True
        else:
            # Get the initialized logger, if not exist,
            # create a logger named `mmcv`
            logger_names = list(logger_initialized.keys())
            logger_name = logger_names[0] if logger_names else 'mmcv'
            for c_m in self.children():
                # print(c_m.__class__.__name__)
                if c_m.__class__.__name__ == 'ModuleList' and c_m[0].__class__.__name__ == 'DetrTransformerEncoder':
                    print_log(f'skip {c_m.__class__.__name__} of ({c_m[0].__class__.__name__}).init_weights()',logger=logger_name)
                else:
                    for m in c_m.modules():
                        # print(m)
                        if hasattr(m, 'weight') and m.weight.dim() > 1:
                            xavier_init(m, distribution='uniform')
            self._is_init = True
            # exit(0)


    def generate_region_pos_and_mask(self, x):
        bs, img_seq, c, h, w = x.shape
        img_num = bs*img_seq

        # region-level mask
        masks = x.new_zeros((img_num, h, w)).to(torch.bool) # False means not mask
        
        # position encoding
        en_pos_embed = self.region_encode_positional_encoding(masks)  # [img_num, embed_dim, h, w]
        en_pos_embed = en_pos_embed.reshape(img_num, self.embed_dims, -1).permute(2, 0, 1) # [h*w, img_num, c]

        masks = masks.reshape(img_num, -1)  #  [img_num, h*w]

        return en_pos_embed, masks
    

    def generate_time_pos_and_mask(self, x):
        bs, img_seq, c, h, w = x.shape
        region_num = h*w
        masks = x.new_zeros((bs, img_seq)).to(torch.bool) # False means not mask
        # only the shape of masks is used during SeqLearnedPositionalEncoding
        encode_pos_embed = self.time_encode_positional_encoding(masks)  # [bs, embed_dim, seq]
        encode_pos_embed = encode_pos_embed.unsqueeze(0).repeat(region_num, 1, 1, 1) # [h*w, bs, embed_dim, seq]
        encode_pos_embed = encode_pos_embed.reshape(region_num*bs, -1, img_seq).permute(2, 0, 1) # [seq, h*w*bs, embed_dim]
        
        masks = _get_causal_mask(img_seq, x.device) # [seq, seq] 
        return encode_pos_embed, masks


    def forward(self, x, patch_mean_pool=True, img_seq_mask=None):
        """
        x: [batch, img_seq, c, h, w]
        """

        batch, img_seq, c, h, w = x.shape
        region_pos_emb, region_mask = self.generate_region_pos_and_mask(x) # [h*w, bs*img_seq, dim], [bs*img_seq, h*w]
        time_pos_emb, time_mask = self.generate_time_pos_and_mask(x) # [img_seq, h*w*bs, dim], [img_seq, img_seq] 

        x = x.reshape(batch*img_seq, c, h, w)
        x = self.input_proj(x) # [bs*img_seq, 256, h, w]
        x = x.reshape(batch, img_seq, self.embed_dims, h*w).permute(3, 0, 1, 2)  #[h*w, bs, img_seq, dim]

        for i in range(self.block_num):
            #==== region-level encoding ====#
            # input: [h*w, batch, img_seq, dim]
            # output: [h*w, batch*img_seq, dim] 
            x = x.reshape(h*w, batch*img_seq, -1) #[h*w, img_num, dim]
            region_x = self.block_region_layers[i](
                                            query=x,  
                                            key=None,
                                            value=None,
                                            query_pos=region_pos_emb,
                                            query_key_padding_mask=region_mask) 
            # [h*w, batch*img_seq, dim] > [h*w, batch, img_seq, dim]
            region_x = region_x.reshape(h*w, batch, img_seq, -1)

            #==== time-level encoding ====#
            # input: [h*w, batch, img_seq, dim]
            # output: [h*w*batch, seq, dim]
            region_x = region_x.reshape(h*w*batch, img_seq, -1).permute(1, 0, 2)  # [seq, h*w*bs, dim]
            if img_seq_mask is None:
                x = self.block_time_layers[i](
                                        query=region_x, 
                                        key=None, 
                                        value=None,
                                        query_pos=time_pos_emb, 
                                        attn_masks=time_mask)  # [seq, h*w*bs, dim]
            else:
                
                time_mask = img_seq_mask.unsqueeze(0).repeat(h*w, 1, 1) # [h*w, batch, seq]
                time_mask = time_mask.reshape(h*w*batch, -1) # [h*w*bs, seq]
                # print('NaviImageRegionTimeBlockTransformerNeck time_mask.size:', time_mask.size())
                x = self.block_time_layers[i](
                                        query=region_x,  # [seq, h*w*bs, dim]
                                        key=None, 
                                        value=None,
                                        query_pos=time_pos_emb, 
                                        query_key_padding_mask=time_mask)  # [h*w*bs, seq]

            x = x.transpose(0, 1) # [h*w*bs, seq, dim]
            x = x.reshape(h*w, batch, img_seq, -1) # [h*w, bs, img_seq, dim]

        output = x
        if patch_mean_pool:
            output = torch.mean(output, dim=0)  # [bs, img_seq, dim]
        # print('NaviImageRegionTimeTransformerNeck output.size:', en_output.size())
        return output
