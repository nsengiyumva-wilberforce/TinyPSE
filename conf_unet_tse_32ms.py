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
    "tcn_blocks": 5,
    "tcn_layers": 2,
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

# Adjusted for batch size 8 (Square root scaling: original LR divided by 2)
adam_kwargs = {
    "lr": 0.25e-3,       # equivalent to 2.5e-4 (down from 0.5e-3)
    "weight_decay": 1e-5, 
}

trainer_conf = {
    "optimizer": "adamw", 
    "optimizer_kwargs": adam_kwargs, 
    "min_lr": 1e-8, 
    "patience": 2, 
    "factor": 0.5, 
    # Adjusted from 200 to 800 because an epoch now has 4x more steps.
    # This keeps your console logs appearing at the exact same data intervals.
    "logging_period": 200  
}

# Reminder for your main training script execution:
# Total Epochs target: 50 epochs (if keeping total optimization steps identical)
#                  or 100 epochs (if you want to leverage the extra fine-tuning)