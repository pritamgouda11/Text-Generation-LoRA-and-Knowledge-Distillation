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

## Problem 1: Low Rank Adaptation (LoRA)

### Introduction

LoRA is a Parameter-Efficient Fine-tuning (PEFT) technique that allows for selective updating of model parameters, reducing computational overhead while maintaining performance.

### Implementation and Results

LoRA was integrated into the GPT-2 model, and the adaptation was tested on the CoLA dataset. The fine-tuned model achieved a balance between computational efficiency and accuracy, with final training and validation accuracies of 70.47% and 69.26% respectively.

## Problem 2: Knowledge Distillation

### Introduction

Knowledge Distillation aims to transfer knowledge from a larger teacher model to a smaller student model, enabling efficient deployment in resource-constrained environments.

### Implementation and Results

An RNN was trained via knowledge distillation from the fine-tuned GPT-2 model. The student model achieved similar validation performance compared to the teacher model, confirming the effectiveness of the distillation process.

## Files Included

- `Report_23754.pdf`: The complete assignment document.
- `question3.ipynb`: Jupyter notebook for part (a) of Problem 3.

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ml-techniques.git
   cd ml-techniques
