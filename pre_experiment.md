# arctic baseline

```sh
python scripts_method/train.py --demo --setup p1 --method arctic_sf --trainsplit train --valsplit minival
```

./logs/demo:arctic_sf_p1-06.05|00:41-a6106773c

# hiera

## classification mode

```sh
python scripts_method/train.py --demo --backbone hiera --setup p1 --method arctic_sf --trainsplit train --valsplit minival
```

./logs/demo:arctic_sf_p1-06.05|12:05-961caefb6

## nonclassification mode

```python scripts_method/train.py --demo --name hiera_feature_mode --backbone hiera --setup p1 --method arctic_sf --trainsplit train --valsplit minival```

./logs/demo:hiera_feature_mode-c55877acf

# crop arc sf

## origin_as_crop

```python scripts_method/train.py --demo --name origin_as_crop --no_crop --setup p1 --method ref_crop_arctic_sf --trainsplit train --valsplit minival --ref_exp_folder=logs/3558f1342 --ref_mode=offline```

./logs/demo:origin_as_crop-3f2f11401

## crop

```python scripts_method/train.py --demo --name right_crop --setup p1 --method ref_crop_arctic_sf --trainsplit train --valsplit minival --ref_exp_folder=logs/3558f1342 --ref_mode=offline```

./logs/demo:right_crop-a67e9fca0

## masked as crop

```/home/lx/anaconda3/envs/arctic_env/bin/python scripts_method/train.py --demo  --name crop_arc_sf_mask --setup p1 --method ref_crop_arctic_sf --trainsplit train --valsplit minival --ref_exp_folder=logs/3558f1342 --ref_crop_folder=logs/3558f1342/masked_render --ref_mode=offline```

log dir:./logs/demo:crop_arc_sf_mask-f90f77eb4

# distri pts arc

```export CUDA_VISIBLE_DEVICES=0,1,2,3```

```/home/lx/anaconda3/envs/arctic_env/bin/python scripts_method/train.py --demo --name arc_pts_dist --num_gpus=4 --pts_h_num=90 --pts_o_num=125 --setup p1 --method pts_arctic_sf --trainsplit train --valsplit minival```

---

maybe main: logs/demo:arc_pts_dist-adc0038d3

convert

```python -m lightning.pytorch.utilities.consolidate_checkpoint path/to/my/checkpoint```

./logs/demo:arc_pts-81771af3f
