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

print("Number of stages: ", len(all_img_features[0]))
print("Number of frames: ", len(all_img_features))
print(all_img_features[0][0].size())
print(all_img_features[0][1].size())
print(all_img_features[0][2].size())
print(all_feat_sizes[0])
# feat_diff_by_stage = [[] for _ in range(len(all_img_features[0]))]
# pos_diff_by_stage = [[] for _ in range(len(all_img_features[0]))]


def compute_diff(num_frames, img_feats, pos_embeds):
    feat_diffs = []
    pos_diffs = []
    for i in tqdm(range(num_frames - 1)):
        j = i + 1
        feat_i = img_feats[i]
        feat_j = img_feats[j]
        assert feat_i.size(1) == 1
        assert feat_j.size(1) == 1
        feat_i = feat_i.unsqueeze(dim=1)
        feat_j = feat_j.unsqueeze(dim=1)
        feat_diff = torch.div(torch.sum((feat_i - feat_j) ** 2), torch.sum(feat_i ** 2))
        feat_diffs.append(feat_diff.item())

        pos_embed_i = pos_embeds[i]
        pos_embed_j = pos_embeds[j]
        assert pos_embed_i.size(1) == 1
        assert pos_embed_j.size(1) == 1
        pos_embed_i = pos_embed_i.unsqueeze(dim=1)
        pos_embed_j = pos_embed_j.unsqueeze(dim=1)
        pos_embed_diff = torch.div(torch.sum((pos_embed_i - pos_embed_j) ** 2), torch.sum(pos_embed_i ** 2))
        pos_diffs.append(pos_embed_diff.item())

    return feat_diffs, pos_diffs


def plot(all_img_features, all_pos_embeds):
    feat_diffs_by_stage = []
    pos_embed_diffs_by_stage = []
    num_stages = len(all_img_features[0])

    for stage in range(num_stages):
        img_feats = [all_img_features[i][stage] for i in range(num_frames)]
        pos_embeds = [all_pos_embeds[i][stage] for i in range(num_frames)]

        feat_diffs, pos_diffs = compute_diff(num_frames, img_feats, pos_embeds)
        feat_diffs_by_stage.append(feat_diffs)
        pos_embed_diffs_by_stage.append(pos_diffs)

    print(min(pos_embed_diffs_by_stage[0]))
    print(max(pos_embed_diffs_by_stage[0]))

    plt.figure(figsize=(30, 10))
    frames = np.arange(len(feat_diffs_by_stage[0]))
    feat_diffs_stage_1 = np.array(feat_diffs_by_stage[0], dtype=float)
    feat_diffs_stage_2 = np.array(feat_diffs_by_stage[1], dtype=float)
    feat_diffs_stage_3 = np.array(feat_diffs_by_stage[2], dtype=float)
    plt.plot(frames, feat_diffs_stage_1, marker='^', markersize=4, label='stage1_img_feat_diff')
    plt.plot(frames, feat_diffs_stage_2, marker='o', markersize=4, label='stage2_img_feat_diff')
    plt.plot(frames, feat_diffs_stage_3, marker='s', markersize=4, label='stage3_img_feat_diff')
    plt.xlabel('frame')
    plt.ylabel('diff')

    plt.legend()
    plt.savefig("feat_diff_stage.pdf")
    plt.close()

    plt.figure(figsize=(30, 10))
    frames = np.arange(len(pos_embed_diffs_by_stage[0]))
    pos_diffs_stage_1 = np.array(pos_embed_diffs_by_stage[0], dtype=float)
    pos_diffs_stage_2 = np.array(pos_embed_diffs_by_stage[1], dtype=float)
    pos_diffs_stage_3 = np.array(pos_embed_diffs_by_stage[2], dtype=float)
    plt.plot(frames, pos_diffs_stage_1, marker='^', markersize=4, label='stage1_pos_embed_diff')
    plt.plot(frames, pos_diffs_stage_2, marker='o', markersize=4, label='stage2_pos_embed_diff')
    plt.plot(frames, pos_diffs_stage_3, marker='s', markersize=4, label='stage3_pos_embed_diff')
    plt.ylim(-0.001, 0.001)
    plt.xlabel('frame')
    plt.ylabel('diff')

    plt.legend()
    plt.savefig("pos_embed_diff_stage.pdf")
    plt.close()


plot(all_img_features, all_pos_embeds)


