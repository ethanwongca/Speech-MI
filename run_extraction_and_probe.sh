#!/bin/bash
#SBATCH --job-name=run_extraction_probe
#SBATCH --account=<account>
#SBATCH --time=12:00:00
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --output=extraction_probe.out
#SBATCH --error=extraction_probe.error
#SBATCH --mail-user=<email>
#SBATCH --mail-type=ALL

module load gcc arrow

source ~/ai_env/bin/activate

echo "Creating Reference Dataset"

python create_reference_dataset.py

echo "Creating Extraction Dataset"

python create_extraction_dataset.py

echo "Creating Probing Dataset"

python create_probing_dataset.py 

echo "Running Probing"

python run_probing.py