GLCONTEXT_LINUX_LIBX11=/usr/lib/x86_64-linux-gnu/libX11.so
GLCONTEXT_LINUX_LIBGL=
GLCONTEXT_GLVERSION=

allo:p1
ego:p2

/targets/targets.mano.pose.r.pt

## visualize

```sh
# ArcticNet-SF-Allo
python scripts_method/extract_predicts.py --setup p1 --method arctic_sf --load_ckpt logs/ffea5bbb6/checkpoints/last.ckpt --run_on val --extraction_mode vis_pose
python scripts_method/visualizer.py --exp_folder logs/ffea5bbb6 --seq_name s05_laptop_use_01_3 --mode gt_mesh --headless
python scripts_method/visualizer.py --exp_folder logs/ffea5bbb6 --seq_name s05_laptop_use_01_3 --mode pred_mesh --headless

# InterField-SF-Allo
# dump predictions
python scripts_method/extract_predicts.py --setup p1 --method field_sf --load_ckpt logs/b789158a7/checkpoints/last.ckpt --run_on val --extraction_mode vis_field

# visualize gt field for right hand
python scripts_method/visualizer.py --exp_folder logs/b789158a7 --seq_name s05_laptop_use_01_3 --mode gt_field_r --headless

# visualize predicted field for left hand
python scripts_method/visualizer.py --exp_folder logs/b789158a7 --seq_name s05_laptop_use_01_3 --mode pred_field_l --headless
```

ArcticNet-SF-Allo:ffea5bbb6
ArcticNet-SF-Ego:5f29ab8a3
ArcticNet-LSTM-Allo:438577402
ArcticNet-LSTM-Ego:9a7012542

InterField-SF-Allo:b789158a7
InterField-SF-Ego:8896849dc
InterField-LSTM-Allo:a0f472d99
InterField-LSTM-Ego:4923c6727

```sh
# ArcticNet-SF-Allo
python scripts_method/extract_predicts.py --setup p1 --method arctic_sf --load_ckpt logs/ffea5bbb6/checkpoints/last.ckpt --run_on val --extraction_mode eval_pose
python scripts_method/evaluate_metrics.py --eval_p logs/ffea5bbb6/eval --split val --setup p1 --task pose

# ArcticNet-LSTM-Allo
python scripts_method/extract_predicts.py --setup p1 --method arctic_lstm --load_ckpt logs/438577402/checkpoints/last.ckpt --run_on val --extraction_mode eval_pose
python scripts_method/evaluate_metrics.py --eval_p logs/438577402/eval --split val --setup p1 --task pose

# ArcticNet-SF-Ego
python scripts_method/extract_predicts.py --setup p2 --method arctic_sf --load_ckpt logs/5f29ab8a3/checkpoints/last.ckpt --run_on val --extraction_mode eval_pose
python scripts_method/evaluate_metrics.py --eval_p logs/5f29ab8a3/eval --split val --setup p2 --task pose

# ArcticNet-LSTM-Ego
python scripts_method/extract_predicts.py --setup p2 --method arctic_lstm --load_ckpt logs/9a7012542/checkpoints/last.ckpt --run_on val --extraction_mode eval_pose
python scripts_method/evaluate_metrics.py --eval_p logs/9a7012542/eval --split val --setup p2 --task pose

# InterField-SF-Allo
python scripts_method/extract_predicts.py --setup p1 --method field_sf --load_ckpt logs/b789158a7/checkpoints/last.ckpt --run_on val --extraction_mode eval_field
python scripts_method/evaluate_metrics.py --eval_p logs/b789158a7/eval --split val --setup p1 --task field

# InterField-LSTM-Allo
python scripts_method/extract_predicts.py --setup p1 --method field_lstm --load_ckpt logs/a0f472d99/checkpoints/last.ckpt --run_on val --extraction_mode eval_field
python scripts_method/evaluate_metrics.py --eval_p logs/a0f472d99/eval --split val --setup p1 --task field

# InterField-SF-Ego
python scripts_method/extract_predicts.py --setup p2 --method field_sf --load_ckpt logs/8896849dc/checkpoints/last.ckpt --run_on val --extraction_mode eval_field
python scripts_method/evaluate_metrics.py --eval_p logs/8896849dc/eval --split val --setup p2 --task field

# InterField-LSTM-Ego
python scripts_method/extract_predicts.py --setup p2 --method field_lstm --load_ckpt logs/4923c6727/checkpoints/last.ckpt --run_on val --extraction_mode eval_field
python scripts_method/evaluate_metrics.py --eval_p logs/4923c6727/eval --split val --setup p2 --task field
```