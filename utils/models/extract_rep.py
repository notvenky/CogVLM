# -*- encoding: utf-8 -*-
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from sat.model.mixins import CachedAutoregressiveMixin
from sat.quantization.kernels import quantize
from sat.model import AutoModel


from utils.utils import chat, llama2_tokenizer, llama2_text_processor_inference, get_image_processor
from utils.models import CogAgentModel, CogVLMModel

# checkpoint = torch.load('mp_rank_00_model_states.pt', map_location='cuda')

class ExtractRep(CogAgentModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input_ids, vision_expert_mask, image_embed_mask, **kwargs):
        cross_inputs = {}
        for k in kwargs:
            if k.startswith('cross_'):
                cross_inputs[k[6:]] = kwargs[k]
        if kwargs.get("mems_cross") is not None:
            encoder_outputs = kwargs["mems_cross"][0]
        else:
            outputs = self.get_mixin('encoder')(**cross_inputs)
            encoder_outputs = outputs
        kwargs['encoder_outputs'] = encoder_outputs
        kwargs['cross_attention_mask'] = cross_inputs['attention_mask']
        
        final_output = super().forward(input_ids=input_ids, vision_expert_mask=vision_expert_mask, image_embed_mask=image_embed_mask, **kwargs)
        
        return final_output, encoder_outputs

    def extract_representations(self, input_ids, vision_expert_mask, image_embed_mask, **kwargs):
        _, representations = self.forward(input_ids, vision_expert_mask, image_embed_mask, **kwargs)
        return representations
    

    # call this inside forward and then do a decode step after this
    # confirm that they have identical outputs
    # seed(42)to reproduce the same results


"""
model_config.json
{
    "model_class": "CogAgentModel",
    "tokenizer_type": "vicuna-7b-v1.5",
    "text_processor_version": "chat",
    "num_layers": 32,
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "vocab_size": 32000,
    "layernorm_order": "pre",
    "model_parallel_size": 1,
    "max_sequence_length": 4096,
    "is_decoder": [
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true
    ],
    "cross_attn_hidden_size": 1024,
    "use_bias": false,
    "inner_hidden_size": 11008,
    "cross_hidden_size_per_attention_head": 32,
    "encoder_args": {},
    "image_length": 256,
    "cross_image_pix": 1120,
    "eva_args": {
        "model_class": "EVA2CLIPModel",
        "num_layers": 63,
        "hidden_size": 1792,
        "num_attention_heads": 16,
        "vocab_size": 1,
        "layernorm_order": "post",
        "model_parallel_size": 1,
        "max_sequence_length": 257,
        "inner_hidden_size": 15360,
        "use_final_layernorm": false,
        "layernorm_epsilon": 1e-06,
        "row_parallel_linear_final_bias": false,
        "image_size": [
            224,
            224
        ],
        "pre_len": 1,
        "post_len": 0,
        "in_channels": 3,
        "patch_size": 14
    },
    "use_neft_noise": true,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "pad_token_id": 0
"""