PEAK_PATH="/Users/alfred/Datasets/ContraNovo/bacillus.10k.mgf"
CKPT_PATH="/Users/alfred/Checkpoints/ContraNovo/ControNovo.ckpt"
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m ContraNovo.ContraNovo  --mode=eval --peak_path="$PEAK_PATH" --model="$CKPT_PATH"