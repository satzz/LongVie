accelerate launch --main_process_port 21455 --config_file accelerate_config_14B.yaml \
    train_longvie_control.py \
    --dataset_base_path data/example_video_dataset \
    --dataset_metadata_path /path/to/your/train_data.json \
    --height 352 \
    --width 640 \
    --dataset_repeat 1 \
    --gradient_accumulation_steps 2 \
    --model_id_with_origin_paths "Wan-AI/Wan2.1-I2V-14B-480P:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-I2V-14B-480P:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-I2V-14B-480P:Wan2.1_VAE.pth,Wan-AI/Wan2.1-I2V-14B-480P:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
    --learning_rate 1e-5 \
    --num_epochs 20 \
    --remove_prefix_in_ckpt "pipe.dual_controller." \
    --output_path "./output/train/LongVie_control" \
    --trainable_models "dual_controller" \
    --data_file_keys 'video,depth,track' \
    --extra_inputs input_image \
    --dataset_num_workers 8 \
    --save_steps 100

# accelerate launch --main_process_port 21455 --config_file accelerate_config_14B.yaml \
#     train_longvie_history_control.py \
#     --dataset_base_path data/example_video_dataset \
#     --dataset_metadata_path /path/to/your/train_data.json \
#     --height 352 \
#     --width 640 \
#     --dataset_repeat 1 \
#     --gradient_accumulation_steps 2 \
#     --model_id_with_origin_paths "Wan-AI/Wan2.1-I2V-14B-480P:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-I2V-14B-480P:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-I2V-14B-480P:Wan2.1_VAE.pth,Wan-AI/Wan2.1-I2V-14B-480P:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
#     --learning_rate 5e-6 \
#     --num_epochs 20 \
#     --remove_prefix_in_ckpt "pipe.dual_controller." \
#     --output_path "./output/train/LongVie_control_hitory" \
#     --trainable_models "dual_controller" \
#     --data_file_keys 'video,depth,track' \
#     --extra_inputs input_image \
#     --dataset_num_workers 8 \
#     --save_steps 100