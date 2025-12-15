import argparse
import numpy as np
import os
import torch
import torch.distributed as dist
from models.vda.video_depth_anything.video_depth import VideoDepthAnything
from models.vda.utils.dc_utils import read_video_frames, vis_sequence_depth, save_video

import decord
from torchvision.transforms import InterpolationMode
import torchvision.transforms as TT


def resize_for_rectangle_crop(arr, image_size, reshape_mode="random"):
    """
    Resize and crop a video tensor to a fixed rectangular resolution.

    Args:
        arr: torch.Tensor of shape [T, C, H, W]
        image_size: [target_height, target_width]
        reshape_mode: "random", "center", or "none"
    """
    if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
        arr = TT.functional.resize(
            arr,
            size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
            interpolation=InterpolationMode.BICUBIC,
        )
    else:
        arr = TT.functional.resize(
            arr,
            size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
            interpolation=InterpolationMode.BICUBIC,
        )

    h, w = arr.shape[2], arr.shape[3]
    arr = arr.squeeze(0)

    delta_h = h - image_size[0]
    delta_w = w - image_size[1]

    if reshape_mode in ["random", "none"]:
        top = np.random.randint(0, delta_h + 1)
        left = np.random.randint(0, delta_w + 1)
    elif reshape_mode == "center":
        top, left = delta_h // 2, delta_w // 2
    else:
        raise NotImplementedError

    arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
    return arr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Depth Anything (Single Video)")
    parser.add_argument("--input_video", type=str, required=True, help="Path to the input video file")
    parser.add_argument("--output_dir",type=str,default="./outputs",help="Directory to save depth results")
    parser.add_argument("--input_size", type=int, default=518)
    parser.add_argument("--max_res", type=int, default=1280)
    parser.add_argument("--encoder", type=str, default="vitl", choices=["vits", "vitl"])
    parser.add_argument("--max_len", type=int, default=-1)
    parser.add_argument("--target_fps", type=int, default=-1)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    video_depth_anything = VideoDepthAnything(**model_configs[args.encoder])
    video_depth_anything.load_state_dict(torch.load(f'./checkpoints/video_depth_anything_{args.encoder}.pth', map_location='cpu'), strict=True)
    video_depth_anything = video_depth_anything.to(DEVICE).eval()


    video_path = args.input_video
    assert os.path.exists(video_path), f"Video not found: {video_path}"
    vr = decord.VideoReader(video_path)
    frame_indices = list(range(len(vr)))
    target_fps = vr.get_avg_fps()

    frames = vr.get_batch(frame_indices).asnumpy()  # [T, H, W, C]
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # [T, C, H, W]
    frames = resize_for_rectangle_crop(frames, image_size=[480, 832], reshape_mode="center")
    frames = frames.permute(0, 2, 3, 1).numpy()  # [T, H, W, C]


    depth_list, fps = video_depth_anything.infer_video_depth(frames, target_fps, input_size=args.input_size, device=DEVICE)
    depth = np.stack(depth_list, axis=0)  # [T, H, W]

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_path = os.path.join(args.output_dir, f"{video_name}_depth.npy")
    np.save(save_path, depth)

    print(f"[✓] Depth saved to: {save_path}")
