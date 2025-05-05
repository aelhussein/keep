# KEEP: Knowledge-preserving and Empirically-refined Embedding Process

This repository contains the implementation code for "KEEP: Integrating Medical Ontologies with Clinical Data for Robust Code Embeddings", a framework that combines knowledge graphs with clinical data to create enhanced medical code embeddings.

## Pipeline Overview

The core workflow involves generating various medical code embeddings and then evaluating them both intrinsically and extrinsically on clinical prediction tasks.

### 1. Data Preparation (Upstream)

While the primary `datasets/` folder is ignored by Git, the script `cohorts_patients/create_cohort_sentence.py` is included. This script appears to be responsible for processing patient data (likely from sources not included in the repository) into sequences suitable for downstream model training and evaluation. Configuration details for data loading during evaluation are found in `extrinsic_evaluation/configs.py` and handled by `extrinsic_evaluation/dataloader.py`.

### 2. Embedding Generation

The repository supports generating several types of medical code embeddings using the following scripts:

*   **KEEP Embeddings (Our Method):**
    *   Generate initial embeddings from knowledge graph structure using `trained_embeddings/our_embeddings/train_node2vec.py`.
    *   Refine these embeddings using clinical co-occurrence data via regularized GloVe training with `trained_embeddings/our_embeddings/train_glove.py`. Set `INIT_EMBEDDING=True` and `REGULARIZATION=True` to enable the KEEP process (using node2vec output as initialization).

*   **Baseline Embeddings:**
    *   **GloVe:** Train standard GloVe embeddings using `trained_embeddings/our_embeddings/train_glove.py` (with `INIT_EMBEDDING=False`).
    *   **Node2Vec:** Generate standard Node2Vec embeddings using `trained_embeddings/our_embeddings/train_node2vec.py`.
    *   **Cui2Vec:** Generate Cui2Vec embeddings using `trained_embeddings/cui2vec/train_cui2vec.py`.
    *   **Pre-trained Language Models:** Generate embeddings from BioClinical BERT, Clinical BERT, and BioGPT using `pretrained_embeddings/get_embeddings.py`. This script supports both standard descriptions and hierarchy-aware description variants.

### 3. Embedding Evaluation

*   **Intrinsic Evaluation:**
    *   Run `intrinsic_evaluation/intrinsic_evaluation.py` to assess how well the generated embeddings capture known medical relationships, hierarchical structures, and co-occurrence patterns present in the data. Helper functions are located in `utils.py`.

*   **Extrinsic Evaluation (Clinical Prediction):**
    *   **Configuration:** Define hyperparameters, embedding paths, and other settings in `extrinsic_evaluation/configs.py`.
    *   **Data Loading:** Prepare data for training/testing using `extrinsic_evaluation/dataloader.py`.
    *   **Model Definition:** The downstream prediction task uses a transformer-based model defined in `extrinsic_evaluation/transformer_model.py`.
    *   **Hyperparameter Tuning:**
        *   Perform learning rate tuning using `extrinsic_evaluation/lr_tuning.py`.
        *   Submit tuning jobs (likely on a cluster like SLURM) using `extrinsic_evaluation/lr_tuning_jobs.sh`.
    *   **Final Training & Evaluation:**
        *   Execute the final model training runs using the best hyperparameters found during tuning with `extrinsic_evaluation/final_training.py`. This script performs multiple runs for statistical analysis.
        *   Submit final training jobs using `extrinsic_evaluation/final_training_jobs.sh`.

## Running the Pipeline

1.  Prepare your input data (knowledge graphs, co-occurrence matrices, patient sequences) as needed by the scripts (Note: the `datasets` directory structure itself is not tracked by git).
2.  Configure paths and parameters within the relevant Python scripts (especially `configs.py` for extrinsic evaluation).
3.  Run the embedding generation scripts (`train_node2vec.py`, `train_glove.py`, etc.) to create the desired embeddings.
4.  Run the intrinsic evaluation script (`intrinsic_evaluation.py`).
5.  Run the extrinsic evaluation pipeline:
    *   Execute `lr_tuning_jobs.sh` (or `lr_tuning.py` directly) for hyperparameter search.
    *   Execute `final_training_jobs.sh` (or `final_training.py` directly) for final model training and evaluation across multiple runs.

*(Note: The `.sh` scripts are likely designed for SLURM or a similar job scheduler and may need adaptation for different environments.)*

## Citation

If you use this code or the KEEP framework in your research, please cite our paper:

```bibtex
@InProceedings{pmlr-v287-elhussein25a,
  title = 	 {{KEEP}: Integrating Medical Ontologies with Clinical Data for Robust Code Embeddings},
  author =       {Elhussein, Ahmed and Meddeb, Paul and Newbury, Abigail and Mirone, Jeanne and Stoll, Martin and G{\"u}rsoy, Gamze},
  booktitle = 	 {Proceedings of the Conference on Health, Inference, and Learning},
  pages = 	 {Accepted},
  year = 	 {2025},
  volume = 	 {287},
  series = 	 {Proceedings of Machine Learning Research},
  publisher =    {PMLR},
}