#!/usr/bin/bash
#SBATCH --mem=16gb                   # Job memory request
#SBATCH --time=0                      # Time limit hrs:min:sec
#SBATCH --gres=gpu:1                # Number of gpu
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --output=./log/%j.log   # Standard output and error log, the program output will be here
​
# you can always have this
eval "$(conda shell.bash hook)"
# you environment
source /home/nnishika/miniconda3/etc/profile.d/conda.sh
conda activate latr
​
export TQDM_DISABLE=1
# code

# image ablation
python3 trainscripts/eval.py "version_576605/epoch=1-step=4954-v1.ckpt" \
    --ablation='image' --split="test" 
# regular
# python3 trainscripts/eval.py "version_577472/epoch=0-step=3303.ckpt"
# python3 trainscripts/eval.py "version_577472/epoch=0-step=3303.ckpt" \
#     --split="test" 
# text ablation
# python3 trainscripts/eval.py "version_577491/epoch=0-step=3303.ckpt" \
#     --ablation='text' --split="test"
# augmented
# python3 trainscripts/eval.py "version_577737/epoch=0-step=2161.ckpt"
# python3 trainscripts/eval.py "version_577662/epoch=0-step=2161.ckpt" \
#     --split="test" 
