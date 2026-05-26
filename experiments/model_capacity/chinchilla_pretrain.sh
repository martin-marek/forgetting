python ~/ptx-mini/train_single.py \
  data.path=/dev/shm/c4_10B/c4_en.bin \
  mix.path=/dev/shm/c4_10B/c4_es.bin \
  mix.coeff=0.3 \
  model.save_dir=~/gcs/ptx-mini/runs/model_capacity_v2_20tpp \
  stop.tokens_per_param=20