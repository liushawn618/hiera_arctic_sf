# Arc

## sf

allo(20epoch): 9d9e425b8
```python scripts_method/train.py --setup p1 --method arctic_sf --trainsplit train --valsplit minival```

```sh
# PLACEHOLDERS
python scripts_method/extract_predicts.py --setup p1 --method arctic_sf --load_ckpt logs/9d9e425b8/checkpoints/last.ckpt --run_on val --extraction_mode eval_pose
python scripts_method/evaluate_metrics.py --eval_p logs/9d9e425b8/eval --split val --setup p1 --task pose
```

logs/9d9e425b8/results

---

ego(100epoch): 5d8710c74

```python scripts_method/train.py --setup p2 --method arctic_sf --trainsplit train --valsplit minival --load_ckpt logs/9d9e425b8/checkpoints/last.ckpt```

# Inter

## sf

allo(20epoch): db8787fbc
```python scripts_method/train.py --setup p1 --method field_sf --trainsplit train --valsplit minival```

```sh
# PLACEHOLDERS
python scripts_method/extract_predicts.py --setup p1 --method field_sf --load_ckpt logs/db8787fbc/checkpoints/last.ckpt --run_on val --extraction_mode eval_field
python scripts_method/evaluate_metrics.py --eval_p logs/db8787fbc/eval --split val --setup p1 --task field

```

logs/db8787fbc/results

ego(100epoch): 559c8479f
```python scripts_method/train.py --setup p2 --method field_sf --trainsplit train --valsplit minival --load_ckpt logs/db8787fbc/checkpoints/last.ckpt```

## lstm

allo():
```python scripts_method/extract_predicts.py --setup p1 --method field_sf --load_ckpt logs/db8787fbc/checkpoints/last.ckpt --run_on train --extraction_mode feat_pose```
