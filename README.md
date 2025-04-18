# Minimalist GPT Implementation from Scratch

This repository contains a minimalist implementation of a Generative Pre-trained Transformer (GPT) model, built from scratch using PyTorch. The goal is to provide a clear and understandable codebase demonstrating the core components of a transformer architecture for text generation.

## Architecture

The model follows the standard GPT architecture, consisting of the following key components:

1.  **Tokenizer**: A simple character-level tokenizer converts the input text into sequences of integer IDs.
2.  **Embedding Layer**: Maps input token IDs to dense vector representations (`d_model`).
3.  **Transformer Blocks**: The core of the model, stacked multiple times (`n_blocks`). Each block contains:
    *   **Multi-Head Self-Attention**: Allows the model to weigh the importance of different tokens in the input sequence when generating the representation for a specific token. It consists of multiple "heads" (`n_heads`), each performing scaled dot-product attention in parallel on a subspace (`dk = d_model / n_heads`).
        *   **Scaled Dot-Product Attention**: The attention mechanism calculates scores based on queries (Q), keys (K), and values (V) derived from the input embeddings. The formula is:
            ```math
            \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
            ```
            A causal mask (lower triangular matrix) is applied before the softmax to prevent positions from attending to subsequent positions.
        *   **Residual Connection & Layer Normalization**: Applied after the attention mechanism.
    *   **Feed-Forward Network (FFN)**: A position-wise fully connected feed-forward network applied independently to each position. It typically consists of two linear layers with a ReLU activation in between.
        ```math
        FFN(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2
        ```
        This implementation uses a configurable number of intermediate layers (`n_layers` in `FeedForward`).
        *   **Residual Connection & Layer Normalization**: Applied after the FFN.
4.  **Final Linear Layer & Softmax**: Projects the output of the last transformer block back to the vocabulary size to produce logits, followed by a Layer Normalization. During generation or loss calculation, a softmax function converts these logits into probabilities.

## Training

*   **Data**: The model is trained on the "Tiny Shakespeare" dataset.
*   **Batching**: The `get_batch` function creates batches of input sequences (`Xs`) and corresponding target sequences (`ys`), where `ys` is `Xs` shifted by one position.
*   **Loss Function**: Cross-Entropy Loss is used to measure the difference between the predicted token probabilities and the actual next tokens.
*   **Optimizer**: Adam optimizer is used for training.
*   **Process**: The `train` method iterates through epochs and batches, performs forward and backward passes, updates model parameters, logs training loss, evaluates validation loss periodically, and saves model checkpoints.

## Usage

1.  **Prerequisites**: Ensure you have Python and PyTorch installed.
2.  **Dataset**: Download or place the `tiny_shakspeare.txt` file in the same directory as `main.py`.
3.  **Run**: Execute the main script:
    ```bash
    python main.py
    ```
    This will initialize the model, tokenizer, and start the training process according to the parameters defined in the script. Training progress, validation loss, and checkpoint saving messages will be printed to the console.

## Configuration Parameters

The key hyperparameters can be adjusted in `main.py`:

*   `CONTEXT_WINDOW`: The length of input sequences.
*   `d_model`: The dimensionality of embeddings and hidden states.
*   `n_heads`: The number of attention heads.
*   `d_ff`: The inner dimension of the Feed-Forward Network.
*   `n_blocks`: The number of transformer blocks.
*   `BATCH_SIZE`: The number of sequences per training batch.
*   Training parameters (`epochs`, `num_batches`, `log_steps`, `val_steps`, `save_steps`, learning rate in `Adam`).

