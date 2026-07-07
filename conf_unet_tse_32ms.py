import CONSTANTS as C
fs = 8000 
chunk_len = 4  # (s)
chunk_size = chunk_len * fs 

nnet_conf = {
    "win_len": 256, 
    "win_inc": 64, 
    "fft_len": 256, 
    "win_type": "sqrthann",
    "kernel_size": (3, 3),
    "stride1": (1, 1), 
    "stride2": (1, 2), 
    "paddings": (1, 0),
    "output_padding": (0, 0),
    "tcn_dims": 64, 
    "tcn_blocks":5,
    "tcn_layers": 1,
    "causal": False,
    "num_spks": 1 
}

train_dir = C.TRAIN_DIR
dev_dir = C.DEV_DIR

train_data = {
    "mix_scp": train_dir + C.TRAIN_MIX,
    "ref_scp": train_dir + C.TRAIN_REF,
    "aux_scp": train_dir + C.TRAIN_AUX,
    "sample_rate": fs,
}


dev_data = { 
    "mix_scp": dev_dir + C.DEV_MIX, 
    "ref_scp": dev_dir + C.DEV_REF,
	"aux_scp": dev_dir + C.DEV_AUX,
    "sample_rate": fs,
}

# trainer config
adam_kwargs = {
    "lr": 0.5e-3, 
    "weight_decay": 1e-5, 
}

trainer_conf = {
    "optimizer": "adamw", 
    "optimizer_kwargs": adam_kwargs, 
    "min_lr": 1e-8, 
    "patience": 2, 
    "factor": 0.5, 
    "logging_period": 200  
}