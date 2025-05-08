# ongoing
## audio-flux
- [ ] rectified flow
- [ ] joint optimization with vocoder https://github.com/Mddct/transformer-vocos

### Data Prepare
```bash
{"wav": "/data/BAC009S0764W0121.wav", 'code': [100, 201, 1, 19]}
{"wav": "/data/BAC009S0764W0122.wav", 'code': [1,20, 101]}
```
### train
```bash
train_data = 'train.jsonl'
model_dir = 'audio_flux/exp/2025/0.1/transformer/'
tensorboard_dir = ${model_dir}/runs/

mkdir -p $model_dir $tensorboard_dir
torchrun --standalone --nnodes=1 --nproc_per_node=8 flux/main.py -- \
        --config vocos/configs/default.py \
        --config.train_data=${train_data} \
        --config.model_dir=${model_dir} \
        --config.tensorboard_dir=${tensorboard_dir} \
        --config.max_train_steps 1000000
```

