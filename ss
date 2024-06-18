```sh
# ArcticNet-SF-Allo
python scripts_method/extract_predicts.py --setup p1 --method arctic_sf --load_ckpt logs/ffea5bbb6/checkpoints/last.ckpt --run_on test --extraction_mode submit_pose
# ArcticNet-LSTM-Allo
python scripts_method/extract_predicts.py --setup p1 --method arctic_lstm --load_ckpt logs/438577402/checkpoints/last.ckpt --run_on test --extraction_mode submit_pose

# ArcticNet-SF-Ego
python scripts_method/extract_predicts.py --setup p2 --method arctic_sf --load_ckpt logs/5f29ab8a3/checkpoints/last.ckpt --run_on test --extraction_mode submit_pose
# ArcticNet-LSTM-Ego
python scripts_method/extract_predicts.py --setup p2 --method arctic_lstm --load_ckpt logs/9a7012542/checkpoints/last.ckpt --run_on test --extraction_mode submit_pose

# InterField-SF-Allo
python scripts_method/extract_predicts.py --setup p1 --method field_sf --load_ckpt logs/b789158a7/checkpoints/last.ckpt --run_on test --extraction_mode submit_field
# InterField-LSTM-Allo
python scripts_method/extract_predicts.py --setup p1 --method field_lstm --load_ckpt logs/a0f472d99/checkpoints/last.ckpt --run_on test --extraction_mode submit_field

# InterField-SF-Ego
python scripts_method/extract_predicts.py --setup p2 --method field_sf --load_ckpt logs/8896849dc/checkpoints/last.ckpt --run_on test --extraction_mode submit_field
# InterField-LSTM-Ego
python scripts_method/extract_predicts.py --setup p2 --method field_lstm --load_ckpt logs/4923c6727/checkpoints/last.ckpt --run_on test --extraction_mode submit_field
```