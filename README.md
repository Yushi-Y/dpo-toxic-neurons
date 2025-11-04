# How Does DPO Reduce Toxicity? A Mechanistic Neuron-Level Analysis

Code for [How Does DPO Reduce Toxicity? A Mechanistic Neuron-Level Analysis](https://aclanthology.org/2025.emnlp-main.1501/), accepted at EMNLP 2025 Main Conference. 
[Video](https://underline.io/lecture/131253-how-does-dpo-reduce-toxicityquestion-a-mechanistic-neuron-level-analysis) | [Slides](https://underline.io/lecture/131253-how-does-dpo-reduce-toxicityquestion-a-mechanistic-neuron-level-analysis) | [Poster](https://underline.io/lecture/131253-how-does-dpo-reduce-toxicityquestion-a-mechanistic-neuron-level-analysis)

![Neuron Groups Visualization](four_neuron_groups.png)

## Overview

We decoded how Direct Preference Optimization (DPO) reduces toxicity in language models through mechanistic interpretability. 

We found that DPO collects activation changes across four neuron groups to reduce toxicity, and simply editing those four neuron groups can replicate DPO's effects! 

This codebase provides scripts for: training linear toxicity probes, training DPO models, identifying the four neuron groups, applying activation patching to validate the four groups, and applying activation editing on four groups to replicate DPO. 


## Installation

```bash
pip install -r requirements.txt
```

## Models and Data

We used the toxicity dataset in [A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and Toxicity](https://arxiv.org/abs/2401.01967) to train linear probes. The datasets are available [here](https://drive.google.com/drive/folders/1baArqcjIc2Q4OllLVUz1hp3p3XxmdteK?usp=drive_link).

Unzip the data files under `toxicity_pairwise`.


## Train Linear Toxicity Probes

To train linear probes for toxicity detection, run:

```bash
python ./toxic_probe/toxic_probe.py
```

## Train DPO Models

To fine-tune an LLM using DPO, run:

```bash
python ./train_dpo/train.py
```

We support training for all huggingface models such as Llama, Gemma, Mistral families.
Configuration files are located in `./train_dpo/config/`. Modify the config files to specify model, training parameters, and hyperparameters. 


## Identify Four Neuron Groups

To compute neuron projections to the linear probes, run:

```bash
python ./activation_analysis/activation_projection.py
```

To identify and group neurons into the four groups (TP-, TN+, AP+, AN-), see:

```bash
./figures/group_neurons_gpt2.ipynb
./figures/group_neurons_llama3.ipynb
```

## Apply Activation Patching to Validate Four Groups

To apply activation patching on the four groups and validate their effects on toxicity, run:

```bash
python ./activation_patching/run_evaluations.py
```

## Apply Activation Editing on Four Groups to Replicate DPO

To apply activation editing on the four neuron groups to replicate DPO's effects, use the evaluation scripts with activation editing hooks:

```bash
python ./activation_patching/run_evaluations.py
```

## Citation

If you find our work useful, please cite it as follows:

```bibtex
@inproceedings{yang-etal-2025-dpo,
    title = "How Does {DPO} Reduce Toxicity? A Mechanistic Neuron-Level Analysis",
    author = "Yang, Yushi  and
      Sondej, Filip  and
      Mayne, Harry  and
      Lee, Andrew  and
      Mahdi, Adam",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.1501/",
    pages = "29512--29531",
}
```
