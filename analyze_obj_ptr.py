import os
import torch
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


root_dir = "./obj_ptrs"
frame_idx = 0

all_obj_ptrs = []
while os.path.exists(os.path.join(root_dir, f"frame{frame_idx}.pt")):
    obj_ptr = torch.load(os.path.join(root_dir, f"frame{frame_idx}.pt"), map_location='cpu')
    all_obj_ptrs.append(obj_ptr)
    frame_idx += 1

num_frames = frame_idx
obj_ptr_diffs = []
for i in range(num_frames - 1):
    j = i + 1
    prev_obj_ptr = all_obj_ptrs[i]
    cur_obj_ptr = all_obj_ptrs[j]
    num_params = prev_obj_ptr.numel()
    obj_ptr_diff = torch.sum((prev_obj_ptr - cur_obj_ptr) ** 2).item() / num_params
    obj_ptr_diffs.append(obj_ptr_diff)


plt.figure(figsize=(30, 10))
frames = np.arange(len(obj_ptr_diffs))
obj_ptr_diffs = np.array(obj_ptr_diffs, dtype=float)
plt.plot(frames, obj_ptr_diffs, marker='o', markersize=4, label='object_pointer_diff')
plt.xlabel('frame')
plt.ylabel('diff')

plt.legend()
plt.savefig("obj_ptr_diff.pdf")
plt.close()
