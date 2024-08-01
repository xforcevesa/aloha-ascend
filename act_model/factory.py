import torch
from modeling_act import ACTPolicy
from configuration_act import ACTConfig
import numpy as np
from torch import optim
from torch.optim import lr_scheduler

demo_config = ACTConfig(
    n_obs_steps=1,
    chunk_size=100,
    n_action_steps=100,
    input_shapes={
        'observation.images.top': [3, 480, 640], 
        'observation.state': [6]
    },
    output_shapes={
        'action': [6]
    },
    input_normalization_modes={
        'observation.images.top': 'mean_std', 
        'observation.state': 'mean_std'
    }, 
    output_normalization_modes={
        'action': 'mean_std'
    }, 
    vision_backbone='resnet18', 
    pretrained_backbone_weights='ResNet18_Weights.IMAGENET1K_V1', 
    replace_final_stride_with_dilation=False, 
    pre_norm=False, 
    dim_model=512, 
    n_heads=8, 
    dim_feedforward=3200, 
    feedforward_activation='relu', 
    n_encoder_layers=4, 
    n_decoder_layers=1, 
    use_vae=True, 
    latent_dim=32, 
    n_vae_encoder_layers=4, 
    temporal_ensemble_coeff=None, 
    dropout=0.1, 
    kl_weight=10.0
)

stats = {
    'observation.images.top': {
        'mean': torch.tensor([0.485, 0.456, 0.406]),
        'std': torch.tensor([0.229, 0.224, 0.225])
    },
    'observation.state': {
        'mean': torch.tensor([0.0]),
        'std': torch.tensor([1.0])
    },
    'action': {
        'mean': torch.tensor([0.0]),
        'std': torch.tensor([1.0])
    }
}


def create_act_model(config=demo_config) -> ACTPolicy:
    return ACTPolicy(config, stats)


def test():
    model = create_act_model()
    state = torch.randn(32, 6)
    action = torch.randn(32, 100, 6)
    top_image = torch.randn(32, 3, 480, 640)
    action_pad = torch.BoolTensor(np.zeros((32, 100)))
    action_pad[:, -1] = True
    batch = {
        'observation.images.top': top_image,
        'observation.state': state,
        'action': action,
        'action_is_pad': action_pad
    }
    output = model.forward(batch)
    print(output)


def train():
    model = create_act_model()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)


if __name__ == '__main__':
    test()

