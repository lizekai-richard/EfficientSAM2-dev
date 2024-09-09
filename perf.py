import os
import torch
import numpy as np
from sam2.build_sam import build_sam2_video_predictor

checkpoint = "./checkpoints/sam2_hiera_base_plus.pt"
model_cfg = "sam2_hiera_b+.yaml"
predictor = build_sam2_video_predictor(model_cfg, checkpoint)


with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    state = predictor.init_state("./video_samples/")

    # add new prompts and instantly get the output on the same frame
    ann_frame_idx = 0
    ann_obj_id = 1
    points = np.array([[210, 350]], dtype=np.float32)
    labels = np.array([1], np.int32)
    frame_idx, object_ids, masks = predictor.add_new_points_or_box(
        inference_state=state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )

    # # propagate the prompts to get masklets throughout the video
    # for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
    #     ...
    
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }