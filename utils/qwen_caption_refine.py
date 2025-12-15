import argparse
import os
import json
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

device = "cuda"
parser = argparse.ArgumentParser(description="Revise captions with Qwen2.5-VL-7B-Instruct")
parser.add_argument("--image_path", type=str, required=True, help="Path to reference image")
args = parser.parse_args()

image_path = args.image_path
json_path = os.path.join(base_dir, "cond.json")
output_file = os.path.join(base_dir, f"new_cond.json")

print(f"[INFO] Using image_path: {image_path}")
print(f"[INFO] Derived json_path: {json_path}")
print(f"[INFO] Derived output_file: {output_file}")


with open(json_path, "r") as f:
    json_data = json.load(f)


model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct",torch_dtype=torch.bfloat16).to(device)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

json_list = []
for video_item in json_data:
    video_path = video_item["video"]
    caption = video_item["text"]
    depth = video_item.get("depth", None)
    track = video_item.get("track", None)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {
                    "type": "text",
                    "text": (
                        "Please revise the following caption so that it accurately matches the visual "
                        "content and semantic context of the given image. Ensure the caption maintains a "
                        "consistent style and tone with other video captions, describing what is happening "
                        "in the image in a natural and coherent way.\n\nCaption: {}".format(caption)
                    )
                },
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        fps=8
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(inputs["input_ids"][0]):] for out_ids in generated_ids
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

    json_list.append({
        "video": video_path,
        "text": output_text[0],
        "depth": depth,
        "track": track,
    })

    with open(output_file, "w") as f:
        json.dump(json_list, f, indent=2)
