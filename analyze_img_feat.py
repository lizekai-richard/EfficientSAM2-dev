import os
import torch
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


feat_dir = "./sample_img_feats"

all_img_features = []
all_pos_embeds = []
all_feat_sizes = []

frame_id = 0
while os.path.exists(os.path.join(feat_dir, f"frame{frame_id}_vision_feats.pt")):
    vision_feats = torch.load(os.path.join(feat_dir, f"frame{frame_id}_vision_feats.pt"), map_location='cpu')
    pos_embeds = torch.load(os.path.join(feat_dir, f"frame{frame_id}_pos_embeds.pt"), map_location='cpu')
    feat_sizes = torch.load(os.path.join(feat_dir, f"frame{frame_id}_feat_sizes.pt"), map_location='cpu')

    all_img_features.append(vision_feats)
    all_pos_embeds.append(pos_embeds)
    all_feat_sizes.append(feat_sizes)

    frame_id += 1

num_frames = frame_id

# feat_diff_by_stage = [[] for _ in range(len(all_img_features[0]))]
# pos_diff_by_stage = [[] for _ in range(len(all_img_features[0]))]


def compute_diff(stage, num_frames, img_feats, pos_embeds):
    feat_diff = []
    pos_diff = []
    for i in tqdm(range(num_frames)):
        for j in range(i):
            feat_i = img_feats[i][stage]
            feat_j = img_feats[j][stage]
            feat_diff = torch.sum((feat_i - feat_j) ** 2) / torch.sum(feat_i ** 2)
            feat_diff.append(feat_diff.item())

            pos_embed_i = pos_embeds[i][stage]
            pos_embed_j = pos_embeds[j][stage]
            pos_embed_diff = torch.sum((pos_embed_i - pos_embed_j) ** 2) / torch.sum(pos_embed_i ** 2)
            pos_diff.append(pos_embed_diff.item())

    return feat_diff, pos_diff


param_list = [(stage, num_frames, all_img_features[stage], all_pos_embeds[stage])
              for stage in range(len(all_img_features[0]))]
with multiprocessing.Pool(processes=len(all_img_features[0])) as pool:
    results = pool.map(compute_diff, param_list)


feat_diff_by_stage = [result[0] for result in results]
pos_embed_diff_by_stage = [result[1] for result in results]


plt.figure(figsize=(20, 8))
frames = np.arange(len(feat_diff_by_stage[0]))
stage_1 = np.array(feat_diff_by_stage[0], dtype=float)
stage_2 = np.array(feat_diff_by_stage[1], dtype=float)
stage_3 = np.array(feat_diff_by_stage[2], dtype=float)

plt.plot(frames, stage_1, marker='^', label='stage1_feat')
plt.plot(frames, stage_2, marker='o', label='stage2_feat')
plt.plot(frames, stage_3, marker='s', label='stage3_feat')

plt.xlabel('frame')
plt.ylabel('diff')
plt.xticks([i for i in range(1, len(frames) + 1, 20)])

plt.legend()
plt.savefig("feat_diff.pdf")
plt.close()


plt.figure(figsize=(20, 8))
frames = np.arange(len(pos_embed_diff_by_stage[0]))
stage_1 = np.array(pos_embed_diff_by_stage[0], dtype=float)
stage_2 = np.array(pos_embed_diff_by_stage[1], dtype=float)
stage_3 = np.array(pos_embed_diff_by_stage[2], dtype=float)

plt.plot(frames, stage_1, marker='^', label='stage1_pos_embed')
plt.plot(frames, stage_2, marker='o', label='stage2_pos_embed')
plt.plot(frames, stage_3, marker='s', label='stage3_pos_embed')

plt.xlabel('frame')
plt.ylabel('diff')
plt.xticks([i for i in range(1, len(frames) + 1, 20)])

plt.legend()
plt.savefig("pos_embed_diff.pdf")
plt.close()


