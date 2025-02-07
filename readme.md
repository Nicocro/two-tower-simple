# Two-Tower Neural Search Engine - Minimal Implementation

## Overview
This repository provides a minimal yet complete implementation of a two-tower neural model for search/retrieval tasks. The goal is to build a search engine that can effectively match queries with relevant documents.

The implementation serves as a starting point for more complex architectures, with the core components designed to handle the Microsoft Machine Reading Comprehension (MS MARCO) dataset - a large-scale dataset specifically designed for search-related deep learning tasks. [link here](https://huggingface.co/datasets/microsoft/ms_marco/viewer)

## Project Structure
The repository is organized as a skeleton following ML project conventions

---

For quick experimentation, check out `minimal-main.py` which contains the complete implementation in a single file, using a dummy dataset.

## Implementation Details
The project leverages PyTorch's core components:
- `torch.nn.Module` for model architecture
- `torch.utils.data.Dataset` for data handling
- `torch.utils.data.DataLoader` for batch processing
- `torch.nn.TripletMarginWithDistanceLoss` for contrastive learning

Key components:
1. Two-Tower Architecture:
   - Query Tower: Processes search queries
   - Passage Tower: Processes document passages
   - Both towers map inputs to a shared embedding space, through common frozen embeddings and individual RNN towers

2. Training Approach:
   - Triplet loss with cosine similarity
   - Each training example consists of (query, positive passage, negative passage)
   - Model learns to embed related query-passage pairs closer together, and push out diverse pairs
