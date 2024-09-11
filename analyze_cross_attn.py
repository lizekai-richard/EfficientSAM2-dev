import os
import torch
import matplotlib.pyplot as plt


cross_attn_dir = "./mem_cross_attn/"
frame_id = 1
all_mem_cross_attns = []
while os.path.exists(os.path.join(cross_attn_dir, f"frame{frame_id}.pt")) and frame_id < 10:
    mem_cross_attn = torch.load(os.path.join(cross_attn_dir, f"frame{frame_id}.pt"), map_location='cpu')
    all_mem_cross_attns.append(mem_cross_attn)

print(len(all_mem_cross_attns[0]))
