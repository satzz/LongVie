import os
import numpy as np
import imageio
from multiprocessing import Pool, cpu_count

src_dir = "./utils/outputs"
dst_root = "./utils/outputs"
os.makedirs(dst_root, exist_ok=True)

def load_and_preprocess(f):
    if "depth" not in f:
        return None
    video_name = f.split("_depth")[0]
    depth = np.load(os.path.join(src_dir, f))
    n_frames = len(depth)
    depth = depth[:1 + 80 * int((n_frames-1)//80)]
    p95, p5 = np.percentile(depth, [95, 5])
    depth = np.clip(depth, p5, p95)
    depth = (p95 - depth) / (p95 - p5)
    return video_name, depth

def save_video(video_name, depth):
    os.makedirs(os.path.join(dst_root, video_name), exist_ok=True)
    n_len = len(depth)

    for i in range(int((n_len-1)//80)):
        start, end = i * 80, i * 80 + 81
        sub_depth = (depth[start:end] * 255).astype(np.uint8)
        sub_depth_rgb = np.stack([sub_depth] * 3, axis=-1)
        out_path = os.path.join(dst_root, video_name, f"depth_{i:02d}.mp4")
        writer = imageio.get_writer(out_path, fps=16, codec="libx264",quality=10, macro_block_size=1)
        for frame in sub_depth_rgb:
            writer.append_data(frame)
        writer.close()
    print(f"[✓] Saved: {video_name}")

if __name__ == "__main__":
    files = os.listdir(src_dir)
    with Pool(processes=min(cpu_count(), 16)) as pool:
        results = pool.map(load_and_preprocess, files)

    for r in results:
        if r is not None:
            save_video(*r)
