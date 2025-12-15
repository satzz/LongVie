import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import decord
from accelerate import Accelerator

from models.spatracker.predictor import SpaTrackerPredictor
from models.spatracker.utils.visualizer import Visualizer

decord.bridge.set_bridge("torch")


# --------------------------------------------------
# Argument parsing
# --------------------------------------------------
parser = argparse.ArgumentParser(description="SpatialTracker (Single Video + Depth)")
parser.add_argument("--video", type=str, required=True, help="Path to RGB video")
parser.add_argument("--depth", type=str, required=True, help="Path to depth video")
parser.add_argument("--save_path", type=str, required=True, help="Output visualization path")
parser.add_argument("--gpu", type=int, default=0)

parser.add_argument("--grid_size", type=int, default=50)
parser.add_argument("--downsample", type=float, default=0.8)
parser.add_argument("--output_fps", type=int, default=8)
parser.add_argument("--len_track", type=int, default=0)
parser.add_argument("--query_frame", type=int, default=0)
parser.add_argument("--point_size", type=int, default=10)
parser.add_argument("--backward", action="store_true")

args = parser.parse_args()


def load_and_preprocess_video(video_path, target_h=480, target_w=832, device="cuda"):
    vr = decord.VideoReader(video_path)
    frames = vr.get_batch(list(range(len(vr))))  # [T,H,W,C]
    video = frames.unsqueeze(0).permute(0, 1, 4, 2, 3).float()  # [1,T,C,H,W]

    _, _, _, H, W = video.shape
    scale = max(target_h / H, target_w / W)

    if scale != 1.0:
        video = F.interpolate(video[0], scale_factor=scale, mode="bilinear", align_corners=True)[None]

    _, _, _, new_H, new_W = video.shape
    sh = (new_H - target_h) // 2
    sw = (new_W - target_w) // 2
    video = video[:, :, :, sh:sh + target_h, sw:sw + target_w]

    segm_mask = np.ones((target_h, target_w), dtype=np.uint8)
    return video.to(device), segm_mask


def load_and_preprocess_depth(depth_path, target_h=480, target_w=832, device="cuda"):
    vr = decord.VideoReader(depth_path)
    depths = vr.get_batch(list(range(len(vr))))  # [T,H,W,C]
    depths = depths.unsqueeze(0).permute(0, 1, 4, 2, 3).float()

    depths = (depths.max() - depths) / (depths.max() - depths.min())
    depths = torch.mean(depths, dim=2).unsqueeze(2)[0]  # [T,1,H,W]

    # resize & crop (same as RGB)
    _, _, H, W = depths.shape
    scale = max(target_h / H, target_w / W)
    if scale != 1.0:
        depths = F.interpolate(depths, scale_factor=scale, mode="bilinear", align_corners=True)

    _, _, new_H, new_W = depths.shape
    sh = (new_H - target_h) // 2
    sw = (new_W - target_w) // 2
    depths = depths[:, :, sh:sh + target_h, sw:sw + target_w]

    return depths.to(device)


def main():
    accelerator = Accelerator()
    device = accelerator.device

    model = SpaTrackerPredictor(
        checkpoint="./checkpoints/spaT_final.pth",
        interp_shape=(384, 576),
        seq_length=12,
    )

    model = accelerator.prepare(model)
    model.eval()

    # Load inputs
    video, segm_mask = load_and_preprocess_video(args.video, device=device)
    depth = load_and_preprocess_depth(args.depth, device=device)

    # Run tracking
    with torch.no_grad():
        pred_tracks, pred_visibility, T_Firsts = model(
            video,
            video_depth=depth,
            grid_size=args.grid_size,
            backward_tracking=args.backward,
            depth_predictor=None,
            grid_query_frame=args.query_frame,
            segm_mask=torch.from_numpy(segm_mask)[None, None].to(device),
            wind_length=12,
            progressive_tracking=False,
        )

    # Visualization
    vis = Visualizer(
        save_dir=os.path.dirname(args.save_path),
        grayscale=False,
        fps=args.output_fps,
        linewidth=args.point_size,
        tracks_leave_trace=args.len_track,
    )

    msk = (T_Firsts == args.query_frame)
    pred_tracks = pred_tracks[:, :, msk.squeeze()]
    pred_visibility = pred_visibility[:, :, msk.squeeze()]

    vis.visualize(
        video=video,
        tracks=pred_tracks,
        visibility=pred_visibility,
        filename=os.path.basename(args.save_path),
        save_path=args.save_path,
    )

    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        print(f"[✓] Saved tracking visualization to {args.save_path}")


if __name__ == "__main__":
    main()
