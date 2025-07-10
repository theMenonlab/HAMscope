#!/bin/bash
#SBATCH --account=notchpeak-gpu
#SBATCH --partition=notchpeak-gpu
#SBATCH --nodes=1
##SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --job-name=training_job_20250425_0gan_single_hs
#SBATCH --output=training_job_20250425_0gan_single_hs%j.log
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=a.ingold@utah.edu

## Activate the conda environment
source activate pix2pix

## Navigate to the project directory
cd probabilistic_hyperspectral_pix2pix_20250401_to_chpc

## Run the training script
python train.py --dataroot /scratch/general/nfs1/u0573922/20250425_dataset_simple/dataset_split --name 20250425_0gan_single_hs --checkpoints_dir /scratch/general/nfs1/u0573922/20250425_dataset_simple/checkpoints --model pix2pix --input_nc 1 --output_nc 60 --n_epochs 15 --n_epochs_decay 15 --save_epoch_freq 5 --netG unet_512 --netG_reps 1 --netD_mult 0 --use_nll 1 --use_comp 3 --use_reg 0 --use_trans 0 --norm layer

python test.py --dataroot /scratch/general/nfs1/u0573922/20250425_dataset_simple/dataset_split --name 20250425_0gan_single_hs --checkpoints_dir /scratch/general/nfs1/u0573922/20250425_dataset_simple/checkpoints --results_dir /scratch/general/nfs1/u0573922/results --model pix2pix --input_nc 1 --output_nc 60 --netG unet_512 --netG_reps 1 --netD_mult 0 --num_test 100 --use_nll 1 --use_comp 3 --use_reg 0 --use_trans 0 --norm layer
