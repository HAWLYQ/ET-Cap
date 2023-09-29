# Copyright (c) AIM3 Lab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, Linear, build_activation_layer
from mmcv.cnn.bricks.transformer import (FFN, build_positional_encoding, build_transformer_layer_sequence)
from mmcv.runner import BaseModule,force_fp32

from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet.models.utils import build_transformer
from ..builder import HEADS, build_loss
from abc import ABCMeta, abstractmethod
from mmdet.datasets.embodiedcap import init_tokenizer
from mmdet.models.decoding_heads.med import (BertConfig, BertLMHeadModel, 
                                            TwoParallelCrossBertLMHeadModel,
                                            TwoCascadedCrossBertLMHeadModel)

def _get_causal_mask(seq_length, device):
    # generate a lower triangular mask, 
    # True means masked item; as torch.nn.MultiheadAttention
    # mask = torch.zeros(seq_length, seq_length, device=device)
    mask = torch.ones(seq_length, seq_length, device=device)
    for i in range(seq_length):
        for j in range(i+1):
            mask[i, j] = 0
    return mask.to(torch.bool)

# refer to BLIP
def load_checkpoint(model, filename):
    checkpoint = torch.load(filename, map_location='cpu') 
    all_state_dict = checkpoint['model']
    # print('model parameters:', model.state_dict().keys())
    # print('pretrained parameters:', all_state_dict.keys())
    # only retrain decoder parameters
    state_dict = {}
    for k, v in all_state_dict.items():
        if 'text_decoder' in k:
            # remove prefx 'text_decoder' to keep consistent with model.state_dict()
            state_dict[k.replace('text_decoder.', '')] = v
            # if k == 'text_decoder.bert.encoder.layer.1.attention.self.value.weight':
            #     print('pretrained weight:', k, v)
    
    for key in model.state_dict().keys():
        if key in state_dict.keys():
            # due to the img feat dim is different with BLIP, part of cross-attention weight can be different
            # crossattention.self.key.weight 768*768
            # crossattention.self.value.weight 768*768
            if state_dict[key].shape!=model.state_dict()[key].shape:
                # print(key, state_dict[key].shape, model.state_dict()[key].shape)
                del state_dict[key]

    msg = model.load_state_dict(state_dict, strict=False)
    print('initialize decoder from %s, missing %d parameters ' % (filename, len(msg.missing_keys)))  
    print('missing parameters:', msg.missing_keys)
    return model, msg


@HEADS.register_module()
class SingleCaptionerHead(BaseModule, metaclass=ABCMeta):
    """Implements the caption transformer head.
    Args:
    """
    def __init__(self,
                encoder_width=256,
                min_dec_len=5,
                max_dec_len=77,
                med_config='configs/single_captioner/bert_config.json',
                dec_beam_size=3,
                repetition_penalty=1.0,
                initialize_from_pretrain=None,
                 **kwargs):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        super(SingleCaptionerHead, self).__init__()
        self.min_dec_len = min_dec_len
        self.max_dec_len = max_dec_len
        self.encoder_width = encoder_width
        self.dec_beam_size = dec_beam_size
        self.repetition_penalty = repetition_penalty
        self.med_config=med_config
        self.initialize_from_pretrain=initialize_from_pretrain

        self.tokenizer = init_tokenizer()
        decoder_config = BertConfig.from_json_file(self.med_config)
        decoder_config.encoder_width = self.encoder_width
        if self.initialize_from_pretrain == 'blip-cap-coco':
            checkpoint_path = '/root/code/BLIP_checkpoints/model_base_caption_capfilt_large.pth'
            self.decoder = BertLMHeadModel(config=decoder_config)
            self.decoder, _ = load_checkpoint(self.decoder, checkpoint_path)
        elif self.initialize_from_pretrain == 'bert-base-uncased':
            self.decoder = BertLMHeadModel.from_pretrained(self.initialize_from_pretrain, config=decoder_config)
            print('initialize decoder from:', self.initialize_from_pretrain)
        else:
            self.decoder = BertLMHeadModel(config=decoder_config)
        self.decoder.resize_token_embeddings(len(self.tokenizer))

    def forward_train(self, image_feats, text_ids, text_mask):
        """Forward function.
            image_feats: [batch_size, h*w, dim]
            text_ids: [batch_size, text_seq]
            text_mask: [batch_size, text_seq]
        """
        """text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=self.max_dec_len, 
                              return_tensors="pt").to(image_feats.device)
        text_mask = text.attention_mask
        text_ids = text.input_ids.clone()  
        text_ids[:,0] = self.tokenizer.bos_token_id""" 
        decoder_targets = text_ids.masked_fill(text_ids == self.tokenizer.pad_token_id, -100)
        # print('SingleCaptionerHead decoder targets:', decoder_targets)
        # print('SingleCaptionerHead text_ids:', text_ids)
        # image_atts = torch.ones(image_feats.size()[:-1],dtype=torch.long).to(image_feats.device)
        image_atts = torch.zeros(image_feats.size()[:-1],dtype=torch.long).to(image_feats.device)
        decoder_output = self.decoder(text_ids, 
                                    attention_mask=text_mask, 
                                    encoder_hidden_states = image_feats,
                                    encoder_attention_mask = image_atts,                  
                                    labels = decoder_targets,
                                    return_dict = True,   
                                    )
        loss_dict = {}
        loss_dict['loss_cap'] = decoder_output.loss
        return loss_dict
    
    def forward_test(self, image_feats):
        # print('SingleCaptionerHead image_feats:', image_feats.size(), image_feats)
        input_ids = torch.tensor([self.tokenizer.bos_token_id]*image_feats.shape[0], device=image_feats.device).unsqueeze(1)
        image_feats = image_feats.repeat_interleave(self.dec_beam_size, dim=0)        
        # image_atts = torch.ones(image_feats.size()[:-1],dtype=torch.long).to(image_feats.device)
        image_atts = torch.zeros(image_feats.size()[:-1],dtype=torch.long).to(image_feats.device)
        # print('SingleCaptionerHead image_atts:', image_atts.size(), image_atts)
        # print('SingleCaptionerHead image_feats:', image_feats.size())
        # print('SingleCaptionerHead input_ids:', input_ids.size())
        # print('SingleCaptionerHead image_atts:', image_atts.size())

        outputs = self.decoder.generate(input_ids=input_ids,
                                        max_length=self.max_dec_len,
                                        min_length=self.min_dec_len,
                                        num_beams=self.dec_beam_size,
                                        eos_token_id=self.tokenizer.sep_token_id,
                                        pad_token_id=self.tokenizer.pad_token_id,     
                                        repetition_penalty=self.repetition_penalty,
                                        encoder_hidden_states=image_feats,
                                        encoder_attention_mask=image_atts)

        # print('SingleCaptionerHead outputs:', outputs)
        captions = []
        for output in outputs:
            caption = self.tokenizer.decode(output, skip_special_tokens=True)
            captions.append(caption)
        # print('SingleCaptionerHead captions:', captions)
        # exit(0)
        return captions

@HEADS.register_module()
class SingleCaptionerTimeHead(SingleCaptionerHead):
    def forward_train(self, time_feats, time_mask, text_ids, text_mask):
        """Forward function.
            time_feats: [batch_size, navigation_seq, dim]
            time_mask: [batch_size, navigation_seq]
            text_ids: [batch_size, text_seq]
            text_mask: [batch_size, text_seq]
        """
        decoder_targets = text_ids.masked_fill(text_ids == self.tokenizer.pad_token_id, -100)
        time_atts = time_mask
        decoder_output = self.decoder(text_ids, 
                                    attention_mask=text_mask, 
                                    encoder_hidden_states = time_feats,
                                    encoder_attention_mask = time_atts,                  
                                    labels = decoder_targets,
                                    return_dict = True,   
                                    )
        loss_dict = {}
        loss_dict['loss_cap'] = decoder_output.loss
        return loss_dict
    
    def forward_test(self, time_feats, time_mask):
        # print('SingleCaptionerHead image_feats:', image_feats.size(), image_feats)
        input_ids = torch.tensor([self.tokenizer.bos_token_id]*time_feats.shape[0], device=time_feats.device).unsqueeze(1)
        time_feats = time_feats.repeat_interleave(self.dec_beam_size, dim=0)        
        # image_atts = torch.ones(image_feats.size()[:-1],dtype=torch.long).to(image_feats.device)
        time_atts = time_mask.repeat_interleave(self.dec_beam_size, dim=0)
        # print('SingleCaptionerHead image_atts:', image_atts.size(), image_atts)
        # print('SingleCaptionerHead image_feats:', image_feats.size())
        # print('SingleCaptionerHead input_ids:', input_ids.size())
        # print('SingleCaptionerHead image_atts:', image_atts.size())

        outputs = self.decoder.generate(input_ids=input_ids,
                                        max_length=self.max_dec_len,
                                        min_length=self.min_dec_len,
                                        num_beams=self.dec_beam_size,
                                        eos_token_id=self.tokenizer.sep_token_id,
                                        pad_token_id=self.tokenizer.pad_token_id,     
                                        repetition_penalty=self.repetition_penalty,
                                        encoder_hidden_states=time_feats,
                                        encoder_attention_mask=time_atts)

        # print('SingleCaptionerHead outputs:', outputs)
        captions = []
        for output in outputs:
            caption = self.tokenizer.decode(output, skip_special_tokens=True)
            captions.append(caption)
        # print('SingleCaptionerHead captions:', captions)
        # exit(0)
        return captions


@HEADS.register_module()
class SingleCaptionerRegionTimeHead(BaseModule, metaclass=ABCMeta):
    """Implements the caption transformer head. 
       each layer contain a self-att and 2 cross-att (region-level and time-level)
    Args:
    """
    def __init__(self,
                encoder_width=256,
                min_dec_len=5,
                max_dec_len=77,
                med_config='configs/single_captioner/2croatt_config.json',
                dec_beam_size=3,
                repetition_penalty=1.0,
                initialize_from_pretrain=None,
                 **kwargs):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        super(SingleCaptionerRegionTimeHead, self).__init__()
        self.min_dec_len = min_dec_len
        self.max_dec_len = max_dec_len
        self.encoder_width = encoder_width
        self.dec_beam_size = dec_beam_size
        self.repetition_penalty = repetition_penalty
        self.med_config=med_config
        self.initialize_from_pretrain=initialize_from_pretrain
        assert self.initialize_from_pretrain is None

        self.tokenizer = init_tokenizer()
        decoder_config = BertConfig.from_json_file(self.med_config)
        decoder_config.encoder_width = self.encoder_width
        # print('SingleCaptionerRegionTimeHead decoder_config.encoder_width:', decoder_config.encoder_width)

        self.decoder = TwoParallelCrossBertLMHeadModel(config=decoder_config)
        self.decoder.resize_token_embeddings(len(self.tokenizer))

    def forward_train(self, region_feats, time_feats, time_mask, text_ids, text_mask):
        """Forward function.
            region_feats: [batch_size, h*w, dim]
            time_feats: [batch_size, navigation_seq, dim]
            time_mask: [batch_size, navigation_seq]
            text_ids: [batch_size, text_seq]
            text_mask: [batch_size, text_seq]
        """
        """text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=self.max_dec_len, 
                              return_tensors="pt").to(image_feats.device)
        text_mask = text.attention_mask
        text_ids = text.input_ids.clone()  
        text_ids[:,0] = self.tokenizer.bos_token_id""" 
        decoder_targets = text_ids.masked_fill(text_ids == self.tokenizer.pad_token_id, -100)
        # print('SingleCaptionerHead decoder targets:', decoder_targets)
        # print('SingleCaptionerHead text_ids:', text_ids)
        # image_atts = torch.ones(image_feats.size()[:-1],dtype=torch.long).to(image_feats.device)
        region_atts = torch.zeros(region_feats.size()[:-1],dtype=torch.long).to(region_feats.device)
        # time_atts = torch.zeros(time_feats.size()[:-1],dtype=torch.long).to(time_feats.device)
        time_atts = time_mask
        decoder_output = self.decoder(text_ids, 
                                    attention_mask=text_mask, 
                                    encoder_hidden_states_1 = region_feats,
                                    encoder_attention_mask_1 = region_atts,
                                    encoder_hidden_states_2 = time_feats,
                                    encoder_attention_mask_2 = time_atts,                  
                                    labels = decoder_targets,
                                    return_dict = True,   
                                    )
        loss_dict = {}
        loss_dict['loss_cap'] = decoder_output.loss
        return loss_dict
    
    def forward_test(self, region_feats, time_feats, time_mask):
        # print('SingleCaptionerHead image_feats:', image_feats.size(), image_feats)
        input_ids = torch.tensor([self.tokenizer.bos_token_id]*region_feats.shape[0], device=region_feats.device).unsqueeze(1)
        region_feats = region_feats.repeat_interleave(self.dec_beam_size, dim=0)    
        time_feats = time_feats.repeat_interleave(self.dec_beam_size, dim=0)

        # image_atts = torch.ones(image_feats.size()[:-1],dtype=torch.long).to(image_feats.device)
        region_atts = torch.zeros(region_feats.size()[:-1],dtype=torch.long).to(region_feats.device)
        # time_atts = torch.zeros(time_feats.size()[:-1],dtype=torch.long).to(time_feats.device)
        time_atts = time_mask.repeat_interleave(self.dec_beam_size, dim=0)
        # print('SingleCaptionerHead image_atts:', image_atts.size(), image_atts)
        # print('SingleCaptionerHead image_feats:', image_feats.size())
        # print('SingleCaptionerHead input_ids:', input_ids.size())
        # print('SingleCaptionerHead image_atts:', image_atts.size())

        outputs = self.decoder.generate(input_ids=input_ids,
                                        max_length=self.max_dec_len,
                                        min_length=self.min_dec_len,
                                        num_beams=self.dec_beam_size,
                                        eos_token_id=self.tokenizer.sep_token_id,
                                        pad_token_id=self.tokenizer.pad_token_id,     
                                        repetition_penalty=self.repetition_penalty,
                                        encoder_hidden_states_1=region_feats,
                                        encoder_attention_mask_1=region_atts,
                                        encoder_hidden_states_2=time_feats,
                                        encoder_attention_mask_2=time_atts,
                                        )

        # print('SingleCaptionerHead outputs:', outputs)
        captions = []
        for output in outputs:
            caption = self.tokenizer.decode(output, skip_special_tokens=True)
            captions.append(caption)
        # print('SingleCaptionerHead captions:', captions)
        # exit(0)
        return captions


@HEADS.register_module()
class SingleCaptionerTimeRegionCascadedHead(BaseModule, metaclass=ABCMeta):
    """Implements the caption transformer head. 
       each layer contain a self-att and 2 cross-att (region-level and time-level)
    Args:
    """
    def __init__(self,
                encoder_width=256,
                min_dec_len=5,
                max_dec_len=77,
                med_config='configs/single_captioner/2croatt_cascaded_config.json',
                dec_beam_size=3,
                repetition_penalty=1.0,
                initialize_from_pretrain=None,
                 **kwargs):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        super(SingleCaptionerTimeRegionCascadedHead, self).__init__()
        self.min_dec_len = min_dec_len
        self.max_dec_len = max_dec_len
        self.encoder_width = encoder_width
        self.dec_beam_size = dec_beam_size
        self.repetition_penalty = repetition_penalty
        self.med_config=med_config
        self.initialize_from_pretrain=initialize_from_pretrain
        assert self.initialize_from_pretrain is None

        self.tokenizer = init_tokenizer()
        decoder_config = BertConfig.from_json_file(self.med_config)
        decoder_config.encoder_width = self.encoder_width
        # print('SingleCaptionerRegionTimeHead decoder_config.encoder_width:', decoder_config.encoder_width)

        self.decoder = TwoCascadedCrossBertLMHeadModel(config=decoder_config)
        self.decoder.resize_token_embeddings(len(self.tokenizer))

    def forward_train(self, region_feats, time_feats, time_mask, text_ids, text_mask):
        """Forward function.
            region_feats: [batch_size, navigation_seq, h*w, dim]
            time_feats: [batch_size, navigation_seq, dim]
            time_mask: [batch_size, navigation_seq]
            text_ids: [batch_size, text_seq]
            text_mask: [batch_size, text_seq]
        """
        """text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=self.max_dec_len, 
                              return_tensors="pt").to(image_feats.device)
        text_mask = text.attention_mask
        text_ids = text.input_ids.clone()  
        text_ids[:,0] = self.tokenizer.bos_token_id""" 
        decoder_targets = text_ids.masked_fill(text_ids == self.tokenizer.pad_token_id, -100)
        # print('SingleCaptionerHead decoder targets:', decoder_targets)
        # print('SingleCaptionerHead text_ids:', text_ids)
        # image_atts = torch.ones(image_feats.size()[:-1],dtype=torch.long).to(image_feats.device)
        bs, time_seq, region_seq, _ = region_feats.size()
        region_atts = torch.zeros([bs, region_seq], dtype=torch.long).to(region_feats.device)
        # time_atts = torch.zeros(time_feats.size()[:-1],dtype=torch.long).to(time_feats.device)
        time_atts = time_mask
        decoder_output = self.decoder(text_ids, 
                                    attention_mask=text_mask, 
                                    encoder_hidden_states_1 = time_feats,
                                    encoder_attention_mask_1 = time_atts,
                                    encoder_hidden_states_2 = region_feats,
                                    encoder_attention_mask_2 = region_atts, 
                                    output_attentions=True, # refer to time-level cross attention weight                 
                                    labels = decoder_targets,
                                    return_dict = True,   
                                    )
        loss_dict = {}
        loss_dict['loss_cap'] = decoder_output.loss
        # TODO: ADD attention weight loss on decoder_output.cross_attentions
        return loss_dict
    
    def forward_test(self, region_feats, time_feats, time_mask):
        bs, time_seq, region_seq, _ = region_feats.size()
        # print('SingleCaptionerHead image_feats:', image_feats.size(), image_feats)
        input_ids = torch.tensor([self.tokenizer.bos_token_id]*region_feats.shape[0], device=region_feats.device).unsqueeze(1)
        region_feats = region_feats.repeat_interleave(self.dec_beam_size, dim=0)    
        time_feats = time_feats.repeat_interleave(self.dec_beam_size, dim=0)

        # image_atts = torch.ones(image_feats.size()[:-1],dtype=torch.long).to(image_feats.device)
        region_atts = torch.zeros([self.dec_beam_size, bs, region_seq],dtype=torch.long).to(region_feats.device)
        # time_atts = torch.zeros(time_feats.size()[:-1],dtype=torch.long).to(time_feats.device)
        time_atts = time_mask.repeat_interleave(self.dec_beam_size, dim=0)
        # print('SingleCaptionerHead image_atts:', image_atts.size(), image_atts)
        # print('SingleCaptionerHead image_feats:', image_feats.size())
        # print('SingleCaptionerHead input_ids:', input_ids.size())
        # print('SingleCaptionerHead image_atts:', image_atts.size())

        outputs = self.decoder.generate(input_ids=input_ids,
                                        max_length=self.max_dec_len,
                                        min_length=self.min_dec_len,
                                        num_beams=self.dec_beam_size,
                                        eos_token_id=self.tokenizer.sep_token_id,
                                        pad_token_id=self.tokenizer.pad_token_id,     
                                        repetition_penalty=self.repetition_penalty,
                                        encoder_hidden_states_1=time_feats,
                                        encoder_attention_mask_1=time_atts,
                                        encoder_hidden_states_2=region_feats,
                                        encoder_attention_mask_2=region_atts,
                                        )

        # print('SingleCaptionerHead outputs:', outputs)
        captions = []
        for output in outputs:
            caption = self.tokenizer.decode(output, skip_special_tokens=True)
            captions.append(caption)
        # print('SingleCaptionerHead captions:', captions)
        # exit(0)
        return captions

    


