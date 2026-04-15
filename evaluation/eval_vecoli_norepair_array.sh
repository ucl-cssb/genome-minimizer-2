#!/bin/bash -l
#$ -l h_rt=2:0:0
#$ -l mem=8G
#$ -l tmpfs=10G
#$ -t 1-100
#$ -N v2nr_VARIANT
#$ -wd /home/ucbt042/Scratch/genome-minimizer
#$ -o /home/ucbt042/Scratch/genome-minimizer/logs/v2nr_VARIANT_$TASK_ID.log
#$ -j y

# vEcoli v2 eval WITHOUT essential gene repair.
# Usage: sed 's/VARIANT/v3/g' eval_v2_norepair_array.sh | qsub

VARIANT=VARIANT
if [ "$SGE_TASK_ID" = "undefined" ] || [ -z "$SGE_TASK_ID" ]; then
    SAMPLE_IDX=0
else
    SAMPLE_IDX=$((SGE_TASK_ID - 1))
fi

echo "=== v2 no-repair eval: $VARIANT sample $SAMPLE_IDX at $(date) ==="

CONTAINER=/home/ucbt042/Scratch/genome-minimizer/python_3.12.12-bookworm.sif
VECOLI=/home/ucbt042/Scratch/genome-minimizer/vEcoli
VENV=/home/ucbt042/Scratch/genome-minimizer/vecoli_venv
DATA=/home/ucbt042/Scratch/genome-minimizer/data
RESULTS=/home/ucbt042/Scratch/genome-minimizer/results_v2/${VARIANT}_no_repair

mkdir -p $RESULTS

apptainer exec --cleanenv \
    --bind $VECOLI:/vEcoli \
    --bind $DATA:/data \
    --bind $RESULTS:/results \
    --bind /home/ucbt042/Scratch/genome-minimizer/evaluation:/eval \
    $CONTAINER bash -c "
source $VENV/bin/activate
export PYTHONPATH=/vEcoli:\$PYTHONPATH

python /eval/eval_single_v2.py \
    --gene-lists /data/${VARIANT}_gene_lists_no_repair.npy \
    --ko-mapping /data/vecoli_knockout_mapping.json \
    --sample-idx $SAMPLE_IDX \
    --sim-data /vEcoli/out/kb/simData.cPickle \
    --output /results \
    --seed 0
"

echo "=== Done: $VARIANT no-repair sample $SAMPLE_IDX at $(date) ==="
