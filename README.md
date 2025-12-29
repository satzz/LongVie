# LongVie 2: Multimodal Controllable Ultra-Long Video World Model

<!-- [![ArXiv](https://img.shields.io/badge/ArXiv-2503.06940-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2503.06940) -->

LongVie 2 is a multimodal controllable world model for generating ultra-long videos with depth and pointmap control signals.

**Authors:**
[Jianxiong Gao](https://jianxgao.github.io/),
[Zhaoxi Chen](https://frozenburning.github.io/),
[Xian Liu](https://alvinliu0.github.io/),
[Junhao Zhuang](https://zhuang2002.github.io/),
[Chengming Xu](https://chmxu.github.io/),
[Jianfeng Feng](https://www.dcs.warwick.ac.uk/~feng/),
[Yu Qiao](https://mmlab.siat.ac.cn/yuqiao/),
[Yanwei Fu†](http://yanweifu.github.io/),
[Chenyang Si†](https://chenyangsi.top/),
[Ziwei Liu†](https://liuziwei7.github.io/)

## 🚀 Quick Start

### Installation
```bash
conda create -n longvie python=3.10 -y
conda activate longvie
conda install psutil
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install ninja
python -m pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.7.2.post1
cd LongVie
pip install -e .
```

### Download Weights

1. Download the base model `Wan2.1-I2V-14B-480P`:
```bash
python download_wan2.1.py
```

2. Download the [LongVie2 weights](https://huggingface.co/Vchitect/LongVie2) and place them in `./models/LongVie/`

### Inference

Generate a 5s video clip (~8-9 mins on a single A100 GPU):
```bash
bash sample_longvideo.sh
```

### Training
```bash
bash train.sh
```

## 🎛️ Control Signal Extraction

We provide utilities for extracting control signals in `./utils`:
```bash
# Extract depth maps
bash get_depth.sh

# Convert depth to .mp4 format
python depth_npy2mp4.py

# Extract trajectory
bash get_track.py
```

To refine prompts after editing the first frame:
```bash
python qwen_caption_refine.py
```

## 📄 Citation

If you find this work useful, please consider citing:
```bibtex
@misc{gao2025longvie,
  title={LongVie: Multimodal-Guided Controllable Ultra-Long Video Generation}, 
  author={Jianxiong Gao and Zhaoxi Chen and Xian Liu and Jianfeng Feng and Chenyang Si and Yanwei Fu and Yu Qiao and Ziwei Liu},
  year={2025},
  eprint={2508.03694},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2508.03694}, 
}

@misc{gao2025longvie2,
  title={LongVie 2: Multimodal Controllable Ultra-Long Video World Model}, 
  author={Jianxiong Gao and Zhaoxi Chen and Xian Liu and Junhao Zhuang and Chengming Xu and Jianfeng Feng and Yu Qiao and Yanwei Fu and Chenyang Si and Ziwei Liu},
  year={2025},
  eprint={2512.13604},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2512.13604}, 
}
```
