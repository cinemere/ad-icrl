DEVICE=cuda \
PYTHONPATH=/content/drive/MyDrive/ml/ad-task/ad-icrl-main \
python3 /content/drive/MyDrive/ml/ad-task/ad-icrl-main/src/dt/train.py \
--config.seq-len 60 \
--config.learning-rate 0.0003 \
--config.weight-decay 0.001 \
--config.embedding-dim 64 \
--config.hidden-dim 256 \
--config.warmup-steps 10000 \
--config.batch-size 128 \
--config.num-workers 10 \
--config.eval-freq 5000 \
--config.eval-episodes 100