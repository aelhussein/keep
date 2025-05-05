# List of outcomes to process
outcomes=(442793 321822 192279 443767 443730 71 66)
outcome_names=("all" "vascular" "kidney" "eye" "neuro")


for outcome in "${outcomes[@]}"; do
    cat << EOF > temp_submit_${outcome}.sh
#!/bin/bash
#SBATCH --job-name=lr_train_${outcome}
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aelhussein@nygenome.org
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --output=logs/outputs/lr_tuning_${outcome}.txt
#SBATCH --error=logs/errors/lr_tuning_${outcome}.txt

# Set environment
ROOT_DIR='/gpfs/commons/projects/ukbb-gursoylab'
WORKING_DIR="\${ROOT_DIR}/aelhussein"

# Activate conda environment
source /gpfs/commons/home/aelhussein/anaconda3/bin/activate cuda_env_ne1

# Run the Python script with the current outcome
python lr_tuning.py --outcome ${outcome} --freeze --load --no-subsample
EOF

    # Submit the job
    sbatch temp_submit_${outcome}.sh
    
    # Remove temporary submission script
    rm temp_submit_${outcome}.sh
    
    echo "Submitted job for outcome ${outcome}"

    sleep 2
done


# for outcome in "${outcomes[@]}"; do
#     cat << EOF > temp_submit_${outcome}.sh
# #!/bin/bash
# #SBATCH --job-name=lr_train_${outcome}_subsample
# #SBATCH --mail-type=ALL
# #SBATCH --mail-user=aelhussein@nygenome.org
# #SBATCH --partition=gpu
# #SBATCH --nodes=1
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=2
# #SBATCH --mem=12G
# #SBATCH --gres=gpu:1
# #SBATCH --time=20:00:00
# #SBATCH --output=logs/outputs/lr_tuning_${outcome}_subsample.txt
# #SBATCH --error=logs/errors/lr_tuning_${outcome}_subsample.txt

# # Set environment
# ROOT_DIR='/gpfs/commons/projects/ukbb-gursoylab'
# WORKING_DIR="\${ROOT_DIR}/aelhussein"

# # Activate conda environment
# source /gpfs/commons/home/aelhussein/anaconda3/bin/activate cuda_env_ne1

# # Run the Python script with the current outcome
# python lr_tuning.py --outcome ${outcome} --freeze --no-load --subsample
# EOF

#     # Submit the job
#     sbatch temp_submit_${outcome}.sh
    
#     # Remove temporary submission script
#     rm temp_submit_${outcome}.sh
    
#     echo "Submitted job for outcome ${outcome}"

#     sleep 2
# done

