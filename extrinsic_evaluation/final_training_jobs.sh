# List of outcomes to process
outcomes=(442793 321822 192279 443767 443730 71 66)
outcome_names=("all" "vascular" "kidney" "eye" "neuro")

for outcome in "${outcomes[@]}"; do
    cat << EOF > temp_submit_${outcome}.sh
#!/bin/bash
#SBATCH --job-name=final_training_${outcome}
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aelhussein@nygenome.org
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH --time=50:00:00
#SBATCH --output=logs/outputs/final_training_${outcome}.txt
#SBATCH --error=logs/errors/final_training_${outcome}.txt

# Set environment
ROOT_DIR='/gpfs/commons/projects/ukbb-gursoylab'
WORKING_DIR="\${ROOT_DIR}/aelhussein"

# Activate conda environment
source /gpfs/commons/home/aelhussein/anaconda3/bin/activate cuda_env_ne1

# Run the Python script with the current outcome
python final_training.py --outcome ${outcome} --freeze --load --overwrite --no-subsample
EOF

    # Submit the job
    sbatch temp_submit_${outcome}.sh
    
    # Remove temporary submission script
    rm temp_submit_${outcome}.sh
    
    echo "Submitted job for outcome ${outcome}"

    sleep 2
done


: <<'BLOCK_COMMENT'
for outcome in "${outcomes[@]}"; do
    cat << EOF > temp_submit_${outcome}.sh
#!/bin/bash
#SBATCH --job-name=final_training_${outcome}_subsample
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aelhussein@nygenome.org
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH --time=30:00:00
#SBATCH --output=logs/outputs/final_training_${outcome}_subsample.txt
#SBATCH --error=logs/errors/final_training_${outcome}_subsample.txt

# Set environment
ROOT_DIR='/gpfs/commons/projects/ukbb-gursoylab'
WORKING_DIR="\${ROOT_DIR}/aelhussein"

# Activate conda environment
source /gpfs/commons/home/aelhussein/anaconda3/bin/activate cuda_env_ne1

# Run the Python script with the current outcome
python final_training.py --outcome ${outcome} --freeze --no-load --no-overwrite --subsample
EOF

    # Submit the job
    sbatch temp_submit_${outcome}.sh
    
    # Remove temporary submission script
    rm temp_submit_${outcome}.sh
    
    echo "Submitted job for outcome ${outcome}"

    sleep 2
done


echo "All jobs submitted"
BLOCK_COMMENT