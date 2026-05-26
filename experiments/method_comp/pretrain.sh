
# pretraining
python ~/ptx-mini/train_single.py \
  data.path=/dev/shm/fineweb_math_4096_10/fineweb_edu_10B.bin \
  model.save_dir=~/gcs/ptx-mini/runs/fwedu \
  opt.peak_lr=1e-3 \
  opt.lr_decay=cosine \
  stop.num_epochs=3 \
  model.D=1024 \
  model.L=16

# finetuning
python ~/ptx-mini/train_single.py \
  data.path=/dev/shm/fineweb_math_4096_10/nemotron_math_20M.bin \
  model.load_path=~/gcs/ptx-mini/runs/fwedu/42gol3bt \
  stop.early_stop_steps=100 \
  opt.peak_lr=1e-5

# lora
model.lora_rank=16

# kl_pre / ntp_pre
reg.method=kl_pre
reg.coeff=10
reg.synth=True
reg.batch_size=64
reg.buffer=16
