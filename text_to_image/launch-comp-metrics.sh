#!/bin/bash
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=diff-models     #Set the job name to "JobExample1"
#SBATCH --time=12:00:00            #Set the wall clock limit to 1hr and 30min
#SBATCH --ntasks=1                 #Request 1 task
#SBATCH --ntasks-per-node=1        #Request 1 task/core per node
#SBATCH --mem=32G               #Request 64GB per node
#SBATCH --gres=gpu:a100:1     #Request 1 GPU
#SBATCH --output=./logs/fk-steering-comp-metrics.%j  #Output file name stdout to [JobID]


cd $SCRATCH/semiblind-dps/Fk-Diffusion-Steering/text_to_image
module load WebProxy
ml Miniconda3
source activate fk-steering

# Set huggingface environment variable
export HF_HOME=$SCRATCH/semiblind-dps/Fk-Diffusion-Steering

python generate_score_hist.py