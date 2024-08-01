import pickle
import numpy as np
import torch
import einops
from tqdm import tqdm

batches = []

for i in range(60):
    print(f'Loading data-{i}.pkl')
    with open(f'output/data-{i}.pkl', 'rb') as f:
        classes = pickle.load(f)

    follower_positions = torch.tensor([dd['follower_pos'] for dd in classes if dd['follower_pos'] is not None])
    # print(follower_positions.shape)
    for idx, dd in enumerate(tqdm(classes)):
        top_image = torch.from_numpy(dd['frame'])
        top_image = einops.rearrange(top_image, 'h w c -> 1 c h w')
        state = follower_positions[idx, :].unsqueeze(0)
        down = 0 if idx < 100 else idx - 99
        action = follower_positions[down: idx + 1, :].unsqueeze(0)
        # print(top_image.shape)
        # print(state.shape)
        # print(action.shape)
        # exit(0)
        batches.append({
            'observation.images.top': top_image,
            'observation.state': state,
            'action': action
        })

with open('output/dataset.pkl', 'wb') as f:
    pickle.dump(batches, f)
    