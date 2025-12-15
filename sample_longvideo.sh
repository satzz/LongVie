export http_proxy=http://10.66.70.227:11080 https_proxy=http://10.66.70.227:11080 no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com
python inference.py \
    --json_file  ./example/ride_horse/cond.json \
    --image_path ./example/ride_horse/first.png  \
    --video_name ride_horse \
    --control_weight_path ./models/LongVie/control.safetensors \
    --dit_weight_path ./models/LongVie/dit.safetensors
