#!/bin/bash -l
#$ -l h_rt=48:0:0
#$ -l mem=8G
#$ -l tmpfs=10G
#$ -t 1-100
#$ -N wcv2_VARIANT
#$ -wd /home/ucbt042/Scratch/genome-minimizer
#$ -o /home/ucbt042/Scratch/genome-minimizer/logs/wcv2_VARIANT_$TASK_ID.log
#$ -j y

# wcEcoli v2 evaluation (correct KO_index + WCM gene set + essential protection).
# Usage: sed 's/VARIANT/v3/g' wcecoli_v2_array.sh | qsub

VARIANT=VARIANT
if [ "$SGE_TASK_ID" = "undefined" ] || [ -z "$SGE_TASK_ID" ]; then
    SAMPLE_IDX=0
else
    SAMPLE_IDX=$((SGE_TASK_ID - 1))
fi

echo "=== wcEcoli v2: $VARIANT sample $SAMPLE_IDX at $(date) ==="

CONTAINER=/home/ucbt042/Scratch/genome-minimizer/wcm-code.sif
DATA=/home/ucbt042/Scratch/genome-minimizer/data
SIMDIR=/home/ucbt042/Scratch/genome-minimizer/wcecoli_out
RESULTS=/home/ucbt042/Scratch/genome-minimizer/results_wcecoli_v2/$VARIANT

mkdir -p $RESULTS

apptainer exec --cleanenv \
    --bind $DATA:/data \
    --bind $SIMDIR:/work \
    --bind $RESULTS:/results \
    --bind /home/ucbt042/Scratch/genome-minimizer/evaluation:/eval \
    --bind $SIMDIR/cache:/wcEcoli/cache \
    $CONTAINER bash -c "
export PYTHONPATH=/wcEcoli
python /eval/wcecoli_eval_v2.py \
    --gene-lists /data/${VARIANT}_gene_lists_with_essentials.npy \
    --wcm-genes /data/BC4_func_genes_indices.csv \
    --essential-genes /data/essential_genes.csv \
    --sample-idx $SAMPLE_IDX \
    --sim-data /work/kb/simData.cPickle \
    --generations 20 \
    --seed 0 \
    --output /results
"

echo "=== Done: $VARIANT sample $SAMPLE_IDX at $(date) ==="
