#!/usr/bin/env python3
"""
Script to generate SLURM batch scripts for hyperspectral pix2pix training jobs.
"""

import os
import re

def parse_filename(filename):
    """Parse filename to extract parameters."""
    # Extract netD_mult (number before 'gan')
    netd_match = re.search(r'(\d+(?:\.\d+)?)gan', filename)
    netd_mult = netd_match.group(1) if netd_match else "0"
    
    # Extract netG_reps (single=1, double=2, tripple=3)
    if 'single' in filename:
        netg_reps = 1
    elif 'double' in filename:
        netg_reps = 2
    elif 'tripple' in filename:
        netg_reps = 3
    else:
        netg_reps = 1  # default
    
    # Determine use_nll and output_nc based on _l1 suffix
    if '_l1' in filename:
        use_nll = 0
        output_nc = 30
    else:
        use_nll = 1
        output_nc = 60
        
    if '_nmf' in filename:
        use_comp = 2
        output_nc = 6
    else:
        use_comp = 3
    
    # Check for _reg in filename
    if '_reg' in filename:
        use_reg = 1
    else:
        use_reg = 0
    
    # Check for _trans in filename
    if '_trans1' in filename:
        use_trans = 1
    else:
        use_trans = 0

    if '_trans2' in filename:
        use_trans = 2
    else:
        use_trans = 0
    
    return netd_mult, netg_reps, use_nll, output_nc, use_comp, use_reg, use_trans

def generate_script(filename):
    """Generate SLURM script content for given filename."""
    netd_mult, netg_reps, use_nll, output_nc, use_comp, use_reg, use_trans = parse_filename(filename)
    
    # Build the command arguments
    train_args = f"--dataroot /scratch/general/nfs1/u0573922/20250425_dataset_simple/dataset_split --name {filename} --checkpoints_dir /scratch/general/nfs1/u0573922/20250425_dataset_simple/checkpoints --model pix2pix --input_nc 1 --output_nc {output_nc} --n_epochs 15 --n_epochs_decay 15 --save_epoch_freq 5 --netG unet_512 --netG_reps {netg_reps} --netD_mult {netd_mult} --use_nll {use_nll} --use_comp {use_comp} --use_reg {use_reg} --use_trans {use_trans} --norm layer"
    
    test_args = f"--dataroot /scratch/general/nfs1/u0573922/20250425_dataset_simple/dataset_split --name {filename} --checkpoints_dir /scratch/general/nfs1/u0573922/20250425_dataset_simple/checkpoints --results_dir /scratch/general/nfs1/u0573922/results --model pix2pix --input_nc 1 --output_nc {output_nc} --netG unet_512 --netG_reps {netg_reps} --netD_mult {netd_mult} --num_test 100 --use_nll {use_nll} --use_comp {use_comp} --use_reg {use_reg} --use_trans {use_trans} --norm layer"
    
    script_content = f"""#!/bin/bash
#SBATCH --account=notchpeak-gpu
#SBATCH --partition=notchpeak-gpu
#SBATCH --nodes=1
##SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --job-name=training_job_{filename}
#SBATCH --output=training_job_{filename}%j.log
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=a.ingold@utah.edu

## Activate the conda environment
source activate pix2pix

## Navigate to the project directory
cd probabilistic_hyperspectral_pix2pix_20250401_to_chpc

## Run the training script
python train.py {train_args}

python test.py {test_args}
"""
    return script_content

def main():
    """Main function to generate all scripts."""
    # Define output directory
    output_dir = "/uufs/chpc.utah.edu/common/home/u0573922/probabilistic_hyperspectral_pix2pix_20250401_to_chpc"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    filenames = [
        "20250425_0gan_single_hs",
        "20250425_0.005gan_single_hs",
        "20250425_0gan_double_hs",
        "20250425_0gan_tripple_hs",
        "20250425_0gan_single_reg_hs",
        "20250425_0gan_single_hs_trans",
        "20250425_0gan_single_hs_trans2",
        "20250425_0gan_single_hs_nmf",
        "20250425_0.005gan_single_hs_nmf",
        "20250425_0gan_double_hs_nmf",
        "20250425_0gan_tripple_hs_nmf",
        "20250425_0gan_single_reg_hs_nmf",
        "20250425_0gan_single_hs_trans_nmf",
        "20250425_0gan_single_hs_trans2_nmf",
        "20250425_0gan_single_hs_1",
        "20250425_0gan_single_hs_2",
        "20250425_0gan_single_hs_3",
        "20250425_0gan_single_hs_4",
        "20250425_0gan_single_hs_5",
        "20250425_0gan_single_hs_l1",
        "20250425_0.005gan_single_hs_l1",
        "20250425_0gan_double_hs_l1",
        "20250425_0gan_tripple_hs_l1",
        "20250425_0gan_single_reg_hs_l1",
        "20250425_0gan_single_hs_trans_l1",
        "20250425_0gan_single_hs_trans2_l1",
    ]
    
    for filename in filenames:
        script_content = generate_script(filename)
        script_filepath = os.path.join(output_dir, f"{filename}.sh")
        
        with open(script_filepath, 'w') as f:
            f.write(script_content)
        
        # Make the script executable
        os.chmod(script_filepath, 0o755)
        
        print(f"Generated: {script_filepath}")
        
        # Print parameters for verification
        netd_mult, netg_reps, use_nll, output_nc, use_comp, use_reg, use_trans = parse_filename(filename)
        print(f"  netD_mult: {netd_mult}, netG_reps: {netg_reps}, use_nll: {use_nll}, output_nc: {output_nc}, use_comp: {use_comp}, use_reg: {use_reg}, use_trans: {use_trans}")

if __name__ == "__main__":
    main()

