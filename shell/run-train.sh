datapath=/home/fei/data/wuhu/ps_images_seg
augpath=/home/fei/code/glass/dtd/images
classes=("dxllt1" 'dgnzj' 'jdx' 'qylgs' 'cxjzq1' 'cxjzq2' 'xjslh1' 'fql' 'sbqb' 'qylgx' 'dxllt4' 'fqg2' 'fqg1' 'dxllt3' 'hxjzq' 'fzzxg' 'zdp' 'hxlg' 'xjslh2' 'cxjzq3' 'rdx' 'dxllt2')
flags=($(for class in "${classes[@]}"; do echo '-d '"${class}"; done))

cd ..
python main_noneed_gt.py \
    --gpu 0 \
    --seed 0 \
    --test train \
  net \
    -b wideresnet50 \
    -le layer2 \
    -le layer3 \
    --pretrain_embed_dimension 1536 \
    --target_embed_dimension 1536 \
    --patchsize 3 \
    --meta_epochs 200\
    --eval_epochs 1 \
    --dsc_layers 2 \
    --dsc_hidden 1024 \
    --pre_proj 1 \
    --mining 1 \
    --noise 0.015 \
    --radius 0.75 \
    --p 0.5 \
    --step 20 \
    --limit 392 \
  dataset \
    --distribution 0 \
    --mean 0.5 \
    --std 0.1 \
    --fg 0 \
    --rand_aug 1 \
    --batch_size 8 \
    --resize 640 \
    --imagesize 640 "${flags[@]}" mvtec $datapath $augpath