import os
import torch
import numpy as np
import matplotlib.pyplot as plt


cross_attn_dir = "./mem_cross_attn/"
frame_id = 1
num_stages = 4
mem_cross_attn_diffs_by_stage = [[] for _ in range(num_stages)]

while os.path.exists(os.path.join(cross_attn_dir, f"frame{frame_id}.pt")) \
        and os.path.exists(os.path.join(cross_attn_dir, f"frame{frame_id + 1}.pt")):
    mem_cross_attn_prev = torch.load(os.path.join(cross_attn_dir, f"frame{frame_id}.pt"), map_location='cuda')
    mem_cross_attn_cur = torch.load(os.path.join(cross_attn_dir, f"frame{frame_id + 1}.pt"), map_location='cuda')

    for stage in range(num_stages):
        cur_stage_prev = mem_cross_attn_prev[stage]
        cur_stage_cur = mem_cross_attn_cur[stage]

        diff = torch.sum((cur_stage_prev - cur_stage_cur) ** 2)
        diff = torch.div(diff, torch.sum(cur_stage_prev ** 2))

        mem_cross_attn_diffs_by_stage[stage].append(diff.item())

    del mem_cross_attn_prev, mem_cross_attn_cur
    torch.cuda.empty_cache()


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


