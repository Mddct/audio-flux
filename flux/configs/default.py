import ml_collections


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.run_name = 'speech_flux'

    config.train_data = ''
    config.eval_data = ''
    config.model_dir = ''
    config.tensorboard_dir = ''
    config.num_workers = 10
    config.prefetch = 100
    config.log_interval = 100
    config.device = 'cuda'
    config.checkpoint = ''

    # train
    # Per device batch size for training.
    config.per_device_batch_size = 32
    # Per device batch size for training.
    config.eval_per_device_batch_size = 32
    config.max_train_steps = 500_000
    config.num_eval_steps = 2_000
    # Base learning rate.
    config.learning_rate = 0.0016
    # Linear learning rate warmup.
    config.warmup_steps = 1000
    # Decay factor for AdamW style weight decay.
    config.weight_decay = 0.1
    # Save a checkpoint every these number of steps.
    config.checkpoint_every_steps = 10_000
    # Frequency of eval during training, e.g. every 1_000 steps.
    config.eval_every_steps = 1_000
    # Use bfloat16 mixed precision training instead of float32.
    config.use_bfloat16 = False
    # Integer for PRNG random seed.
    config.seed = 2025

    # mel
    config.sample_rate = 24000
    config.hop_size = 480  # sample_rate // hop_size = 50 for flow
    config.n_fft = 1920  # hop_size * 4
    config.n_mels = 100  # 128 for future
    config.power = 1
    config.fmin = 0
    config.fmax = None
    config.norm = 'slaney'
    config.mel_scale = 'slaney'
    config.padding = "same"
    config.multiscale_mel_loss = True
    config.clip_grad_norm = 1

    config.speech_vocab_size = 8000
    # model
    # TODO(Mddct): change later when trainer is done
    config.output_size = 256
    config.attention_heads = 4
    config.linear_units = 2048
    config.num_blocks = 12
    config.dropout_rate = 0.0
    config.positional_dropout_rate = 0.0
    config.attention_dropout_rate = 0.0
    config.input_layer = 'linear'
    config.normalize_before = True
    config.query_bias = True
    config.key_bias = True
    config.value_bias = True
    config.activation_type = "relu"
    config.gradient_checkpointing = False
    config.use_sdpa = False
    config.layer_norm_type = "rms_norm"
    config.norm_eps = 1e-5
    config.n_kv_head = None
    config.head_dim = config.output_size // config.attention_heads
    config.selfattention_layer_type = "selfattn"
    config.mlp_type = "moe"
    config.mlp_bias = True
    config.n_expert = 8
    config.n_expert_activated = 2
    config.right_context = 2  # 2*6*20 =240ms
    config.left_context = 15  # 300ms

    return config
