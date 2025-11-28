#!/bin/bash
#SBATCH --job-name=run_extraction_probe
#SBATCH --account=<account>
#SBATCH --time=12:00:00
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --output=extraction_probe.out
#SBATCH --error=extraction_prob.error
#SBATCH --mail-user=<user>
#SBATCH --mail-type=ALL

module load gcc arrow

source ~/ai_env/bin/activate

cd /home/ewong/links/scratch/Speech-MI

echo "Running the extraction script"

python run_extraction_sharded.py

echo "Completed extraction script"

echo "Running the probing script"

python run_probing_sharded.py 

echo "Process is completed"

