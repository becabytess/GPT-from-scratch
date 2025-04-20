# Reinventing ChatGPT: A Foundational Language Model Implementation

## Overview

This project presents a foundational implementation of a Generative Pre-trained Transformer (GPT) model. It focuses on constructing the core components of modern large language models from fundamental principles, including tokenization, the transformer architecture, and the training process.

## Core Components

1.  **Custom Byte-Pair Encoding (BPE) Tokenizer:**
    *   A bespoke tokenizer ([`Tokenizer`](tokenizer.py)) is implemented and trained directly on the target corpus ([tiny_shakspeare.txt](tiny_shakspeare.txt)).
    *   The training process ([`Tokenizer.train`](tokenizer.py) in [train_tokenizer.py](train_tokenizer.py)) iteratively merges frequent byte pairs to build an efficient subword vocabulary from scratch.
    *   The trained tokenizer state is persisted ([tokenizer.pt](tokenizer.pt)) for consistent encoding ([`Tokenizer.encode`](tokenizer.py)) and decoding ([`Tokenizer.decode`](tokenizer.py)).

2.  **GPT Architecture:**
    *   The core model ([`GPT`](main.py)) implements the transformer architecture based on the original "Attention Is All You Need" paper.
    *   Key components include:
        *   Input Embeddings combined with Sinusoidal Positional Encodings ([`GPT.forward`](main.py)).
        *   Multiple stacked Transformer Blocks ([`TransformerBlock`](main.py)), each containing:
            *   Scaled Dot-Product Multi-Head Self-Attention ([`MultiHeadAttention`](main.py)).
            *   Position-wise Feed-Forward Networks ([`FeedForward`](main.py)).
            *   Layer Normalization and Residual Connections applied at standard positions within the block.
        *   A final linear projection layer followed by Softmax for next-token probability distribution.

3.  **Training Framework:**
    *   The training loop ([`GPT.train`](main.py)) manages the model optimization process.
    *   Utilizes Cross-Entropy Loss ([`GPT.__init__`](main.py)) as the objective function.
    *   Employs the Adam optimizer with configurable learning rate scheduling ([`GPT.__init__`](main.py)).
    *   Implements gradient accumulation ([`GPT.train`](main.py)) to effectively increase batch size without exceeding memory constraints.
    *   Includes periodic evaluation on a validation set ([`GPT.evaluate`](main.py)) and model checkpointing ([`GPT.train`](main.py), [`GPT.state_dict`](main.py)) to save training progress ([checkpoints/](checkpoints/), [gpt_latest.pt](gpt_latest.pt)).

## Setup and Usage

1.  **Environment:** Install necessary Python packages.
    ```sh
    pip install -r requirements.txt
    ```

2.  **Train Tokenizer:** Generate the vocabulary and merge rules from the corpus.
    ```sh
    python train_tokenizer.py
    ```
    This process creates the `tokenizer.pt` file.

3.  **Train GPT Model:** Execute the main training script.
    ```sh
    python main.py
    ```
    This script loads the pre-trained `tokenizer.pt`, prepares the dataset ([`get_samples`](main.py)), initializes the [`GPT`](main.py) model, and commences training. Checkpoints are saved periodically in the [checkpoints/](checkpoints/) directory, and the final model state is saved to [gpt_latest.pt](gpt_latest.pt).

## Project Scope

This implementation explores the fundamental building blocks of large language models, providing a practical, code-level understanding of the concepts driving modern generative AI.