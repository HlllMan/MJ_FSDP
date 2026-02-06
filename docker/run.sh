docker run -d -it --rm \
--name miaoji_fsdp_env \
--gpus all \
-p 80:80 \
-v /public/data0/HOME/jdnlp1004/miaoji.norman/FSDP/final:/workspace \
miaoji_torch_2.10:v1 \
bash 

