import os
import torch
import matplotlib.pyplot as plt


feat_dir = "./sample_img_feats"

all_img_features = []
all_pos_embeds = []
all_feat_sizes = []

frame_id = 0
while os.path.exists(os.path.join(feat_dir, f"frame{frame_id}_vision_feats.pt")):
    vision_feat = torch.load(os.path.join(feat_dir, f"frame{frame_id}_vision_feats.pt"))
    pos_embeds = torch.load(os.path.join(feat_dir, f"frame{frame_id}_pos_embeds.pt"))
    feat_sizes = torch.load(os.path.join(feat_dir, f"frame{frame_id}_feat_sizes.pt"))

    all_img_features.append(vision_feats)
    all_pos_embeds.append(pos_embeds)
    all_feat_sizes.append(feat_sizes)

    frame_id += 1