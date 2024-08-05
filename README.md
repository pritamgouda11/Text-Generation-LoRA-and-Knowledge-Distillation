# Machine Learning Techniques: Text Generation, LoRA, and Knowledge Distillation

## Assignment #2 - E0270: Machine Learning

**Author:** Pritam Trilochan Gouda  
**Affiliation:** CSA, IISc  
**Date:** April 27, 2024

---

## Overview

This repository contains the solutions and discussions for Assignment #2 of the course E0270: Machine Learning. The assignment covers three main topics: Text Generation with GPT-2, Low Rank Adaptation (LoRA), and Knowledge Distillation.

## Table of Contents

- [Introduction](#introduction)
- [Problem 0: Text Generation with GPT-2](#problem-0-text-generation-with-gpt-2)
- [Problem 1: Low Rank Adaptation (LoRA)](#problem-1-low-rank-adaptation-lora)
- [Problem 2: Knowledge Distillation](#problem-2-knowledge-distillation)
- [Files Included](#files-included)
- [Plots](#Plots)
- [Usage](#usage)
- [Conclusion](#conclusion)
- [References](#references)

## Introduction

This assignment explores the application and analysis of advanced machine learning techniques, focusing on text generation, efficient model adaptation, and knowledge transfer between models.

## Problem 0: Text Generation with GPT-2

### Overview

An exploration of GPT-2's text generation capabilities was conducted by providing a prompt and analyzing the model's ability to generate a coherent and creative continuation.

### Generated Text Instance

The model generated a narrative based on the prompt, demonstrating its grasp of context and creative storytelling.

<img width="733" alt="Screenshot 2024-07-11 at 12 55 19 PM" src="https://github.com/pritamgouda11/Text-Generation-LoRA-and-Knowledge-Distillation/assets/46958858/0aca474c-0b59-4b25-aede-5b5b592d3fce">

## Problem 1: Low Rank Adaptation (LoRA)

### Introduction

LoRA is a Parameter-Efficient Fine-tuning (PEFT) technique that allows for selective updating of model parameters, reducing computational overhead while maintaining performance.

### Implementation and Results

LoRA was integrated into the GPT-2 model, and the adaptation was tested on the CoLA dataset. The fine-tuned model achieved a balance between computational efficiency and accuracy.

#### Model Details

- **GPT2 Variant Used:** Medium
- **Total Number of Parameters:** 356.40M
- **Number of Trainable Parameters:** 1.68M
- **Reduction in Parameters:** 99.53%
- **Maximum Accuracy on CoLA Validation Dataset:** 82.73%

- **GPT2 Variant Used:** Base
- **Total Number of Parameters:** 125.03M
- **Number of Trainable Parameters:** 0.63M
- **Reduction in Parameters:** 99.50%

#### Training Strategy

The GPT-2 model was fine-tuned using the following hyperparameters:
- **Learning Rate:** 1e-3
- **Number of Epochs:** 10
- **Batch Size:** 128
- **Optimizer:** Adam
- **Loss Function:** Cross-Entropy Loss
- **LoRA Rank:** 4


## Problem 2: Knowledge Distillation

### Introduction

Knowledge Distillation aims to transfer knowledge from a larger teacher model to a smaller student model, enabling efficient deployment in resource-constrained environments.

### Implementation and Results

An RNN was trained via knowledge distillation from the fine-tuned GPT-2 model. The student model achieved similar validation performance compared to the teacher model, confirming the effectiveness of the distillation process.

#### Distillation Strategy

To distill knowledge from the fine-tuned GPT model (teacher model) to the DistilRNN model (student model) for the CoLA classification dataset, the distillation loss function used is a combination of soft target loss and true label loss.

#### DistilRNN Architecture

- Embedding layer mapping input tokens to dense vectors of size 768.
- Two-layer RNN with hidden size 768.
- ReLU activation function.
- Linear layer projecting the output to a 2-dimensional space for binary classification.

#### Optimal Training Hyperparameters

- **Batch size:** 128
- **Learning rate:** 1e-3
- **Number of epochs:** 5

#### Results

- **Maximum Accuracy on CoLA Validation Dataset:** 71%
- **Accuracy without KD:** 68%
- **Accuracy Improvement with KD:** 3%

## Files Included
```
Text Generation, LoRA, and Knowledge Distillation
├── plots
│   ├── Distillation_accuracy.png
│   ├── Distillation_loss.png
│   ├── LoRA_accuracy.png
│   ├── LoRA_loss.png
│   ├── rnn_accuracy.png
│   └── rnn_loss.png
├── tuning
│   ├── tuning.txt
│   ├── tuning2.txt
│   ├── tuning3.txt
│   └── tuning4.txt
├── Report_23754.pdf
├── model.py
├── run.py
├── train_utils.py
└── utils.py
```
- `model.py`: Full definition of a GPT Language Model, all of it in this single file.
- `Report_23754.pdf`: Detailed project report with explanation.


## Plots

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/bc7081c4-d647-4b60-98fa-a0cd4329b404" alt="distil_22582_acc" style="width: 100%;"></td>
    <td><img src="https://github.com/user-attachments/assets/db50a2ce-b0bc-4bbc-bf9f-771b5e837793" alt="distil_22582_loss" style="width: 100%;"></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/7e90829e-0f6a-4c60-afdf-388384c0cfe2" alt="LoRA_22582_acc" style="width: 100%;"></td>
    <td><img src="https://github.com/user-attachments/assets/f745f117-44cc-41f0-a0e9-a40d44bacfe8" alt="LoRA_22582_loss" style="width: 100%;"></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/c5b1c6d9-a520-40c5-adbc-3396a2c953bb" alt="rnn_22582_acc" style="width: 100%;"></td>
    <td><img src="https://github.com/user-attachments/assets/c99ab75e-f43f-474f-8d02-44af1f7b626d" alt="rnn_22582_loss" style="width: 100%;"></td>
  </tr>
</table>



## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ml-techniques.git
   cd ml-techniques
   
## Conclusion

This assignment provides a comprehensive study of advanced ML techniques, demonstrating their practical applications and effectiveness in different scenarios.

## References

[Practical Tips for Finetuning LLMs Using LoRA
](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)

[Knowledge Distillation: Principles, Algorithms, Applications
](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)

[Pretraining a 124-M Parameter GPT-2 Language Model
](https://wandb.ai/bkkaggle/lm-finetuning/reports/Pretraining-a-124-M-Parameter-GPT-2-Language-Model--VmlldzoyMjg4NzA)
