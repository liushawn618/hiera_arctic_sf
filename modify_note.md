## pts arc sf

```sh
python scripts_method/train.py --setup p1 --method pts_arctic_sf --trainsplit train --valsplit minival
```

./logs/52522e680

---

shutil
OSError: [Errno 39] Directory not empty: '/tmp/pymp-_h92esgf'

```export CUDA_VISIBLE_DEVICES=0,1,2,3```

```/home/lx/anaconda3/envs/arctic_env/bin/python scripts_method/train.py --name arc_pts_dist --num_gpus=4 --pts_h_num=90 --pts_o_num=125 --setup p1 --method pts_arctic_sf --trainsplit train --valsplit minival```

arc_pts_dist-0d0915dbd

### eval

running

```sh
# PLACEHOLDERS
/home/lx/anaconda3/envs/arctic_env/bin/python scripts_method/extract_predicts.py --setup p1 --method pts_arctic_sf --load_ckpt logs/52522e680/checkpoints/last.ckpt --run_on val --extraction_mode eval_pose
/home/lx/anaconda3/envs/arctic_env/bin/python scripts_method/evaluate_metrics.py --eval_p logs/52522e680/eval --split val --setup p1 --task pose
```

### result

```json
{
    "aae": 6.292478084564209,
    "mpjpe/ra/h": 23.19853973388672,
    "mrrpe/r/l": 47.702239990234375,
    "mrrpe/r/o": 39.30699157714844,
    "success_rate/0.05": 63.6391453770859,
    "cdev/ho": 42.739383697509766,
    "mdev/h": 9.675946235656738,
    "acc/h": 6.434141159057617,
    "acc/o": 8.128876686096191
}
```

## arc transformer

```sh
/home/lx/anaconda3/envs/arctic_env/bin/python scripts_method/train.py --setup p1 --method arctic_tf --trainsplit train --valsplit minival
```

./logs/ef7e470d5

## ref if sf

### train

```python scripts_method/train.py --setup=p1 --method=ref_field_sf --trainsplit=train --valsplit=minival --ref_exp_folder=logs/3558f1342 --ref_mode=online --ref_setup=p1 --ref_method=arctic_sf --ref_ckpt=logs/3558f1342/checkpoints/last.ckpt```

./logs/2beb0ea84

| loss__val | epoch | step |
| 0.13212400674819946 | 0 | 23381 |
| 0.10842986404895782 | 1 | 46763 |
| 0.08345738053321838 | 2 | 70145 |
| 0.07232138514518738 | 3 | 93527 |
| 0.06959231942892075 | 4 | 116909 |
| 0.06611057370901108 | 5 | 140291 |
| 0.06364835053682327 | 6 | 163673 |
| 0.058466363698244095 | 7 | 187055 |
| 0.05752287805080414 | 8 | 210437 |
| 0.05847363546490669 | 9 | 233819 |
| 0.0540546253323555 | 10 | 257201 |
| 0.0557607039809227 | 11 | 280583 |
| 0.05549610033631325 | 12 | 303965 |
| 0.05323651432991028 | 13 | 327347 |
| 0.05451031029224396 | 14 | 350729 |
| 0.052866678684949875 | 15 | 374111 |
| 0.053678959608078 | 16 | 397493 |
| 0.05283339321613312 | 17 | 420875 |
| 0.05208452045917511 | 18 | 444257 |
| 0.052525486797094345 | 19 | 467639 |


### eval

```sh
# PLACEHOLDERS
python scripts_method/extract_predicts.py --setup=p1 --method=ref_field_sf --load_ckpt="logs/2beb0ea84/checkpoints/last.ckpt" --run_on=val --extraction_mode=eval_field --ref_exp_folder=logs/3558f1342 --ref_mode=online --ref_setup=p1 --ref_method=arctic_sf --ref_ckpt=logs/3558f1342/checkpoints/last.ckpt
python scripts_method/evaluate_metrics.py --eval_p ./logs/2beb0ea84/eval --split val --setup p1 --task field
```

### contrast

epoch 18

```json
{
    "avg/ho": 8.86762523651123,
    "avg/oh": 9.143204689025879,
    "acc/oh": 3.2960970401763916,
    "acc/ho": 2.907723903656006
}
```

last.ckpt

```json
{
    "avg/ho": 8.614195823669434,
    "avg/oh": 9.02243423461914,
    "acc/oh": 3.249160051345825,
    "acc/ho": 2.8897933959960938
}
```

---

no ref: b789158a7

```json
{
    "avg/ho": 8.985878944396973,
    "avg/oh": 9.264034271240234,
    "acc/oh": 2.7591757774353027,
    "acc/ho": 2.856574058532715
}
```
### visualize

#### dump predictions

python scripts_method/extract_predicts.py --setup=p1 --method=ref_field_sf --load_ckpt="logs/2beb0ea84/checkpoints/last.ckpt" --run_on=val --extraction_mode=vis_field --ref_exp_folder=logs/3558f1342 --ref_mode=online --ref_setup=p1 --ref_method=arctic_sf --ref_ckpt=logs/3558f1342/checkpoints/last.ckpt

#### visualize gt field for right hand

python scripts_method/visualizer.py --exp_folder logs/2beb0ea84 --seq_name s05_box_grab_01_3 --mode gt_field_r --headless

#### visualize predicted field for left hand
python scripts_method/visualizer.py --exp_folder logs/2beb0ea84 --seq_name s05_laptop_use_01_3 --mode pred_field_r --headless


## ref if ego

### train

```python scripts_method/train.py --setup=p2 --method=ref_field_sf --trainsplit=train --valsplit=minival --load_ckpt="logs/2beb0ea84/checkpoints/last.ckpt" --ref_exp_folder=logs/3558f1342 --ref_mode=online --ref_setup=p1 --ref_method=arctic_sf --ref_ckpt=logs/3558f1342/checkpoints/last.ckpt```

100 epoch
./logs/efe17a163

### eval

```sh
python scripts_method/extract_predicts.py --setup=p2 --method=ref_field_sf --load_ckpt="logs/efe17a163/checkpoints/last.ckpt" --run_on=val --extraction_mode=eval_field --ref_exp_folder=logs/3558f1342 --ref_mode=online --ref_setup=p1 --ref_method=arctic_sf --ref_ckpt=logs/3558f1342/checkpoints/last.ckpt
python scripts_method/evaluate_metrics.py --eval_p ./logs/efe17a163/eval --split val --setup p2 --task field
```

### result

```json
{
    "avg/ho": 9.442403793334961,
    "avg/oh": 11.018698692321777,
    "acc/oh": 5.063972473144531,
    "acc/ho": 3.435750722885132
}
```
compare
```json
{
    "avg/ho": 8.69062328338623,
    "avg/oh": 9.255694389343262,
    "acc/oh": 2.249075412750244,
    "acc/ho": 2.3838300704956055
}
```

## crop arc sf

```python scripts_method/train.py --setup p1 --method ref_crop_arctic_sf --trainsplit train --valsplit minival --ref_exp_folder=logs/3558f1342 --ref_mode=offline```

./logs/d1eef5cd4
./logs/3f154364f
### eval

```sh
python scripts_method/extract_predicts.py --setup p1 --method ref_crop_arctic_sf  --ref_exp_folder=logs/3558f1342 --ref_mode=offline --load_ckpt="logs/d1eef5cd4/checkpoints/last.ckpt" --run_on=val --extraction_mode=eval_pose
python scripts_method/evaluate_metrics.py --eval_p ./logs/d1eef5cd4/eval --split val --setup p1 --task pose
```

### result

```json
{
    "aae": 44.861385345458984,
    "mpjpe/ra/h": 128.1846160888672,
    "mrrpe/r/l": 506.7619323730469,
    "mrrpe/r/o": 230.359619140625,
    "success_rate/0.05": 2.373238135463269,
    "cdev/ho": 368.21826171875,
    "mdev/h": 11.572225570678711,
    "acc/h": 1.4887912273406982,
    "acc/o": 0.8471505045890808
}
```

### rerun(separated hand)

./logs/3f154364f

#### formal

./logs/ref_crop_arctic_sf_p1_06.04|21:38-e76d6f315

## arc sf allo(hiera)

```sh
python scripts_method/train.py --setup p1 --method arctic_sf --trainsplit train --valsplit minival
```

./logs/8838a894e
```sh
python scripts_method/train.py --name hiera_feature_mode --backbone hiera --setup p1 --method arctic_sf --trainsplit train --valsplit minival
```
eval
```sh
python scripts_method/extract_predicts.py --name hiera_feature_mode --backbone hiera --setup p1 --method arctic_sf --load_ckpt logs/hiera_feature_mode-339bff58a/checkpoints/last.ckpt --run_on val --extraction_mode eval_pose

python scripts_method/evaluate_metrics.py --eval_p logs/hiera_feature_mode-339bff58a/eval --split val --setup p1 --task pose
```
{
    "aae": 5.477935791015625,
    "mpjpe/ra/h": 20.253524780273438,
    "mrrpe/r/l": 41.265560150146484,
    "mrrpe/r/o": 33.04442596435547,
    "success_rate/0.05": 75.05448284447613,
    "cdev/ho": 36.487815856933594,
    "mdev/h": 8.064141273498535,
    "acc/h": 5.302588939666748,
    "acc/o": 7.613159656524658
}
compare
{
    "aae": 5.9705681800842285,
    "mpjpe/ra/h": 22.672298431396484,
    "mrrpe/r/l": 46.53743362426758,
    "mrrpe/r/o": 37.931697845458984,
    "success_rate/0.05": 72.82110133154778,
    "cdev/ho": 39.378421783447266,
    "mdev/h": 9.762165069580078,
    "acc/h": 6.541830539703369,
    "acc/o": 8.338079452514648
}
```sh
python scripts_method/extract_predicts.py --name hiera_feature_mode --backbone hiera --setup p1 --method arctic_sf --load_ckpt logs/hiera_feature_mode-339bff58a/checkpoints/epoch=17-step=420876.ckpt --run_on val --extraction_mode eval_pose

python scripts_method/evaluate_metrics.py --eval_p logs/hiera_feature_mode-339bff58a/eval --split val --setup p1 --task pose
```
{
        "aae": 5.311713218688965,
    "mpjpe/ra/h": 20.22608757019043,
    "mrrpe/r/l": 41.31230545043945,
    "mrrpe/r/o": 34.2510986328125,
    "success_rate/0.05": 74.5233852685957,
    "cdev/ho": 36.76839828491211,
    "mdev/h": 8.270157814025879,
    "acc/h": 5.348310470581055,
    "acc/o": 7.792338848114014
}
./logs/hiera_feature_mode-339bff58a