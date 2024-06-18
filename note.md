# note

visualize:
先terminal ：Xvfb :99
再vscode：export DISPLAY=:99
再运行代码 要用headledudsadasdfdnvdsvdnssd

## visualize

python scripts_method/evaluate_metrics.py --eval_p logs/3558f1342/eval --split val --setup p1 --task pose

python scripts_method/extract_predicts.py --setup p1 --method arctic_sf --load_ckpt logs/3558f1342/checkpoints/last.ckpt --run_on val --extraction_mode vis_pose
python scripts_method/extract_predicts.py --setup p1 --method arctic_sf --load_ckpt logs/3558f1342/checkpoints/last.ckpt --run_on train --extraction_mode vis_pose

## monitor

find logs/3558f1342/eval -type d | wc -l
find logs/3558f1342/eval -type d -name "meta_info" | wc -l

check visualizable scenses

```sh
echo `find logs/3558f1342/eval -type d -name "meta_info" | wc -l`/`find logs/3558f1342/eval -maxdepth 1 -type d | wc -l`
```

visualize all

```python multi_render.py --exp_folder logs/ffea5bbb6 -n 30 --mode gt_mesh --headless --no_model --render_type rgb,mask```

check progress

```sh
echo `ls -l logs/3558f1342/render | wc -l`/`ls -l logs/3558f1342/eval | wc -l`
```

## train

init:
```sh
conda activate arctic_env # source ~/anaconda3/bin/activate arctic_env
export DISPLAY=:99
export CUDA_VISIBLE_DEVICES=0
export COMET_API_KEY="qSRtNsfNMEuAZnp8XWVUbK8zJ"
export COMET_WORKSPACE="liushawn618"
```

### Arc

#### SF:

- Allo:a
    position:ffea5bbb6
    tmux:train_a
```sh
python scripts_method/train.py --setup p1 --method arctic_sf --trainsplit train --valsplit minival
```

Ego:bbbbbbb
    cmd: 
```sh
python scripts_method/train.py --setup p2 --method arctic_sf --trainsplit train --valsplit minival --load_ckpt logs/ffea5bbb6/checkpoints/last.ckpt
```
    position:5f29ab8a3

ArcticNet-LSTM:
Allo:ccccc
    total epochs:10, log dir:./logs/438577402
    train:
```sh
python scripts_method/extract_predicts.py --setup p1 --method arctic_sf --load_ckpt logs/ffea5bbb6/checkpoints/last.ckpt --run_on train --extraction_mode feat_pose
```
    eval:
```sh
python scripts_method/extract_predicts.py --setup p1 --method arctic_sf --load_ckpt logs/ffea5bbb6/checkpoints/last.ckpt --run_on val --extraction_mode feat_pose


```
    packaging:
```sh
python scripts_method/build_feat_split.py --split train --protocol p1 --eval_p logs/ffea5bbb6/eval
python scripts_method/build_feat_split.py --split minitrain --protocol p1 --eval_p logs/ffea5bbb6/eval
python scripts_method/build_feat_split.py --split val --protocol p1 --eval_p logs/ffea5bbb6/eval
python scripts_method/build_feat_split.py --split tinyval --protocol p1 --eval_p logs/ffea5bbb6/eval
python scripts_method/build_feat_split.py --split minival --protocol p1 --eval_p logs/ffea5bbb6/eval
```

ArcticNet-LSTM:
Ego:ddddd
    80epoch, 9a7012542
```sh
python scripts_method/extract_predicts.py --setup p2 --method arctic_sf --load_ckpt logs/5f29ab8a3/checkpoints/last.ckpt --run_on train --extraction_mode feat_pose
python scripts_method/extract_predicts.py --setup p2 --method arctic_sf --load_ckpt logs/5f29ab8a3/checkpoints/last.ckpt --run_on val --extraction_mode feat_pose

```

    train:
```sh
python scripts_method/extract_predicts.py --setup p2 --method arctic_sf --load_ckpt logs/5f29ab8a3/checkpoints/last.ckpt --run_on train --extraction_mode feat_pose
```
    eval:
```sh
python scripts_method/extract_predicts.py --setup p2 --method arctic_sf --load_ckpt logs/5f29ab8a3/checkpoints/last.ckpt --run_on val --extraction_mode feat_pose
```
    packaging:
```sh
python scripts_method/build_feat_split.py --split train --protocol p2 --eval_p logs/5f29ab8a3/eval
python scripts_method/build_feat_split.py --split minitrain --protocol p2 --eval_p logs/5f29ab8a3/eval
python scripts_method/build_feat_split.py --split val --protocol p2 --eval_p logs/5f29ab8a3/eval
python scripts_method/build_feat_split.py --split tinyval --protocol p2 --eval_p logs/5f29ab8a3/eval
python scripts_method/build_feat_split.py --split minival --protocol p2 --eval_p logs/5f29ab8a3/eval
```

### IF

#### sf

InterField-SF:
Allo:eeeee
    position:b789158a7
    train_cmd: 
```sh
python scripts_method/train.py --setup p1 --method field_sf --trainsplit train --valsplit minival
```
Ego:ffffff
    cmd: python scripts_method/train.py --setup p2 --method field_sf --trainsplit train --valsplit minival --load_ckpt logs/b789158a7/checkpoints/last.ckpt
    position: 8896849dc

#### lstm

IF-Allo:ggggg
    position:a0f472d99
    train_cmd: 
```sh
python scripts_method/extract_predicts.py --setup p1 --method field_sf --load_ckpt logs/b789158a7/checkpoints/last.ckpt --run_on train --extraction_mode feat_pose
```
    eval_cmd: 

```sh
python scripts_method/extract_predicts.py --setup p1 --method field_sf --load_ckpt logs/b789158a7/checkpoints/last.ckpt --run_on val --extraction_mode feat_pose
```

    packaging: 
    
```sh
python scripts_method/build_feat_split.py --split train --protocol p1 --eval_p logs/b789158a7/eval
python scripts_method/build_feat_split.py --split minitrain --protocol p1 --eval_p logs/b789158a7/eval
python scripts_method/build_feat_split.py --split val --protocol p1 --eval_p logs/b789158a7/eval
python scripts_method/build_feat_split.py --split tinyval --protocol p1 --eval_p logs/b789158a7/eval
python scripts_method/build_feat_split.py --split minival --protocol p1 --eval_p logs/b789158a7/eval
```

IF-Ego:hhhhh
    position:4923c6727
    train_cmd:
```sh
python scripts_method/extract_predicts.py --setup p2 --method field_sf --load_ckpt logs/8896849dc/checkpoints/last.ckpt --run_on train --extraction_mode feat_pose
```
    eval_cmd:
```sh
python scripts_method/extract_predicts.py --setup p2 --method field_sf --load_ckpt logs/8896849dc/checkpoints/last.ckpt --run_on val --extraction_mode feat_pose
```
    packaging:
```sh
python scripts_method/build_feat_split.py --split train --protocol p2 --eval_p logs/8896849dc/eval
python scripts_method/build_feat_split.py --split minitrain --protocol p2 --eval_p logs/8896849dc/eval
python scripts_method/build_feat_split.py --split val --protocol p2 --eval_p logs/8896849dc/eval
python scripts_method/build_feat_split.py --split tinyval --protocol p2 --eval_p logs/8896849dc/eval
python scripts_method/build_feat_split.py --split minival --protocol p2 --eval_p logs/8896849dc/eval
```

## eval

```sh
# PLACEHOLDERS
python scripts_method/extract_predicts.py --setup p1 --method arctic_sf --load_ckpt logs/3558f1342/checkpoints/last.ckpt --run_on val --extraction_mode eval_pose
python scripts_method/evaluate_metrics.py --eval_p logs/3558f1342/eval --split val --setup p1 --task pose
```