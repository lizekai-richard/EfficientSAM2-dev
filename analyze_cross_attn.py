import os
import torch
import numpy as np
import matplotlib.pyplot as plt


cross_attn_dir = "./mem_cross_attn/"
frame_id = 1
num_stages = 4
mem_cross_attn_diffs_by_stage = [[] for _ in range(num_stages)]
all_mem_cross_attns = []

while os.path.exists(os.path.join(cross_attn_dir, f"frame{frame_id}.pt")):
    mem_cross_attn = torch.load(os.path.join(cross_attn_dir, f"frame{frame_id}.pt"), map_location='cpu')
    all_mem_cross_attns.append(mem_cross_attn)
    frame_id += 1

num_frames = frame_id - 1
for stage in range(num_stages):
    for i in range(num_frames - 1):
        j = i + 1
        mem_cross_attn_prev = all_mem_cross_attns[i][stage]
        mem_cross_attn_cur = all_mem_cross_attns[j][stage]

        diff = torch.sum((mem_cross_attn_cur - mem_cross_attn_prev) ** 2)
        diff = torch.div(diff, torch.sum(mem_cross_attn_prev ** 2))

        mem_cross_attn_diffs_by_stage[stage].append(diff.item())

plt.figure(figsize=(30, 10))
frames = np.arange(len(mem_cross_attn_diffs_by_stage[0]))
ca_diffs_stage_1 = np.array(mem_cross_attn_diffs_by_stage[0], dtype=float)
ca_diffs_stage_2 = np.array(mem_cross_attn_diffs_by_stage[1], dtype=float)
ca_diffs_stage_3 = np.array(mem_cross_attn_diffs_by_stage[2], dtype=float)
ca_diffs_stage_4 = np.array(mem_cross_attn_diffs_by_stage[3], dtype=float)
plt.plot(frames, ca_diffs_stage_1, marker='^', markersize=4, label='stage1_mem_ca_diff')
plt.plot(frames, ca_diffs_stage_2, marker='o', markersize=4, label='stage2_mem_ca_diff')
plt.plot(frames, ca_diffs_stage_3, marker='s', markersize=4, label='stage3_mem_ca_diff')
plt.plot(frames, ca_diffs_stage_4, marker='*', markersize=4, label='stage4_mem_ca_diff')
plt.xlabel('frame')
plt.ylabel('diff')

plt.legend()
plt.savefig("mem_cross_attn_diff_stage.pdf")
plt.close()


