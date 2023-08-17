import torch
import torch.nn as nn
import pdb

from torchvision.models.resnet import resnet50
from torchvision.models.vgg import vgg16

import dino.vision_transformer as vits

def get_model(arch, patch_size, device):
    if "ibot" in arch:
        # Currently only supporting base
        assert patch_size == 16
        model = vits.__dict__['vit_base'](patch_size=patch_size, return_all_tokens=True)

    else:
        model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)

    for p in model.parameters():
        p.requires_grad = False

    # Initialize model with pretraining
    if "imagenet" not in arch:
        url = None
        if arch == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
            # url = None
        elif arch == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"
        elif arch == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif arch == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        elif arch == "ibot_base" and patch_size == 16:
            url = None
        if url is not None:
            print(
                "Since no pretrained weights have been provided, we load the reference pretrained DINO weights."
            )
            state_dict = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/dino/" + url
            )
            strict_loading = True
            msg = model.load_state_dict(state_dict, strict=strict_loading)
            print(
                "Pretrained weights found at {} and loaded with msg: {}".format(
                    url, msg
                )
            )
        else:
            if arch == "ibot_base":
                state_dict = torch.load('weights/checkpoint_teacher.pth')['state_dict']
                state_dict = {k.replace("module.", ""): v for k, v in
                              state_dict.items()}
            else:
                state_dict = torch.load('weights/checkpoint.pth')['teacher']
                to_remove = ['head.mlp.0.weight', 'head.mlp.0.bias',
                             'head.mlp.2.weight', 'head.mlp.2.bias',
                             'head.mlp.4.weight', 'head.mlp.4.bias',
                             'head.last_layer.weight_g',
                             'head.last_layer.weight_v']
                state_dict = {k.replace('backbone.',''): v for k, v in
                              state_dict.items() if k not in to_remove}

            strict_loading = False if 'ibot' in arch else True
            msg = model.load_state_dict(state_dict, strict=strict_loading)

            print(
                "Pretrained weights found at {} and loaded with msg: {}".format(
                    'weights/checkpoint.pth', msg
                )
            )

    model.eval()
    model.to(device)
    return model

