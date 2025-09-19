#!/bin/bash
EGENES=data/essential_genes/essential_gene_positions.pkl
#python main.py --mode preprocess

python main.py --mode training --preset v0 --epochs 10
python main.py --mode sample --model-path models/trained_models/v0_model/saved_VAE_v0.pt --genes-path $EGENES --num-samples 1000

python main.py --mode training --preset v1 --epochs 10
python main.py --mode sample --model-path models/trained_models/v1_model/saved_VAE_v1.pt --genes-path $EGENES --num-samples 1000

python main.py --mode training --preset v2 --epochs 10
python main.py --mode sample --model-path models/trained_models/v2_model/saved_VAE_v2.pt --genes-path $EGENES --num-samples 1000