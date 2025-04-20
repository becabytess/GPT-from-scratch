# Reinventing ChatGPT: A Foundational Language Model Implementation

## Overview

This project represents a first-principles implementation of a Generative Pre-trained Transformer (GPT) model. It moves beyond leveraging high-level libraries to construct the core components of modern large language models from the ground up, demonstrating a sophisticated understanding of the underlying mechanisms, including tokenization, transformer architecture, and training dynamics.

## Core Components

1.  **Custom Byte-Pair Encoding (BPE) Tokenizer:**
    *   A bespoke tokenizer ([`Tokenizer`](tokenizer.py)) is implemented and trained on the target corpus ([tiny_shakspeare.txt](tiny_shakspeare.txt)).
    *   The training process ([`Tokenizer.train`](tokenizer.py) in [train_tokenizer.py](train_tokenizer.py)) iteratively merges frequent byte pairs to build an efficient subword vocabulary.
    *   The trained tokenizer state is persisted ([tokenizer.pt](tokenizer.pt)) for consistent encoding ([`Tokenizer.encode`](tokenizer.py)) and decoding ([`Tokenizer.decode`](tokenizer.py)).

2.  **GPT Architecture:**
    *   The core model ([`GPT`](main.py)) implements the transformer architecture.
    *   Key components include:
        *   Input Embeddings with Sinusoidal Positional Encodings ([`GPT.forward`](main.py)).
        *   Multiple Transformer Blocks ([`TransformerBlock`](main.py)), each containing:
            *   Multi-Head Self-Attention ([`MultiHeadAttention`](main.py)).
            *   Position-wise Feed-Forward Networks ([`FeedForward`](main.py)).
            *   Layer Normalization and Residual Connections.
        *   A final linear projection layer and Softmax for token prediction.

3.  **Training Framework:**
    *   The training loop ([`GPT.train`](main.py)) orchestrates the learning process.
    *   Utilizes Cross-Entropy Loss ([`GPT.__init__`](main.py)) for optimization.
    *   Employs the Adam optimizer with learning rate scheduling ([`GPT.__init__`](main.py)).
    *   Implements gradient accumulation ([`GPT.train`](main.py)) to simulate larger batch sizes.
    *   Includes periodic validation ([`GPT.evaluate`](main.py)) and model checkpointing ([`GPT.train`](main.py), [`GPT.state_dict`](main.py)) to save progress ([checkpoints/](checkpoints/), [gpt_latest.pt](gpt_latest.pt)).

## Setup and Usage

1.  **Environment:** Install dependencies.
    ```sh
    pip install -r requirements.txt
    ```

2.  **Train Tokenizer:** Generate the vocabulary and merges from the corpus.
    ```sh
    python train_tokenizer.py
    ```
    This creates `tokenizer.pt`.

3.  **Train GPT Model:** Run the main training script.
    ```sh
    python main.py
    ```
    This loads `tokenizer.pt`, prepares the dataset ([`get_samples`](main.py)), and trains the [`GPT`](main.py) model, saving checkpoints in [checkpoints/](checkpoints/) and the latest model state to [gpt_latest.pt](gpt_latest.pt).

## Vision

This project serves as a testament to advanced AI engineering capabilities, showcasing the ability to not only utilize but also construct and understand the intricate details of state-of-the-art language models. It reflects the fundamental research and engineering prowess required for developing next-generation AI systems.