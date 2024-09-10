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
while os.path.exists(os.path.join(feat_dir, f"frame{frame_id}_vision_feats.pt")) and frame_id <= 300:
    vision_feats = torch.load(os.path.join(feat_dir, f"frame{frame_id}_vision_feats.pt"), map_location='cpu')
    pos_embeds = torch.load(os.path.join(feat_dir, f"frame{frame_id}_pos_embeds.pt"), map_location='cpu')
    feat_sizes = torch.load(os.path.join(feat_dir, f"frame{frame_id}_feat_sizes.pt"), map_location='cpu')

    all_img_features.append(vision_feats)
    all_pos_embeds.append(pos_embeds)
    all_feat_sizes.append(feat_sizes)

    frame_id += 1

num_frames = frame_id

print("Number of stages: ", len(all_img_features[0]))
# feat_diff_by_stage = [[] for _ in range(len(all_img_features[0]))]
# pos_diff_by_stage = [[] for _ in range(len(all_img_features[0]))]


def compute_diff(stage, num_frames, img_feats, pos_embeds):
    feat_diffs = []
    pos_diffs = []
    for i in tqdm(range(num_frames)):
        for j in range(i):
            feat_i = img_feats[i]
            feat_j = img_feats[j]
            feat_diff = torch.sum((feat_i - feat_j) ** 2) / torch.sum(feat_i ** 2)
            feat_diffs.append(feat_diff.item())

            pos_embed_i = pos_embeds[i]
            pos_embed_j = pos_embeds[j]
            pos_embed_diff = torch.sum((pos_embed_i - pos_embed_j) ** 2) / torch.sum(pos_embed_i ** 2)
            pos_diffs.append(pos_embed_diff.item())

    return feat_diffs, pos_diffs

def plot(all_img_features, all_pos_embeds):
    for stage in range(len(all_img_features[0])):
        img_feats = [all_img_features[i][stage] for i in range(num_frames)]
        pos_embeds = [all_pos_embeds[i][stage] for i in range(num_frames)]

        feat_diffs, pos_diffs = compute_diff(stage, num_frames, img_feats, pos_embeds)

        plt.figure(figsize=(20, 8))
        frames = np.arange(len(feat_diffs))
        feat_diffs = np.array(feat_diffs, dtype=float)

        plt.plot(frames, feat_diffs, marker='^', label='stage1_img_feat')

        plt.xlabel('frame')
        plt.ylabel('diff')
        plt.xticks([i for i in range(1, len(frames) + 1, 20)])

        plt.legend()
        plt.savefig(f"feat_diff_stage{stage}.pdf")
        plt.close()


        plt.figure(figsize=(20, 8))
        frames = np.arange(len(pos_diffs))
        pos_diffs = np.array(pos_diffs, dtype=float)

        plt.plot(frames, pos_diffs, marker='^', label='stage1_pos_embed')

        plt.xlabel('frame')
        plt.ylabel('diff')
        plt.xticks([i for i in range(1, len(frames), 20)])

        plt.legend()
        plt.savefig(f"pos_embed_diff_stage{stage}.pdf")
        plt.close()


plot(all_img_features, all_pos_embeds)


