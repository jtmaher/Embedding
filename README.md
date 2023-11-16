Banach-Tarski Embeddings
------------------------

This repo contains an implementation of Banach-Tarski embeddings,
which allow us to map recursive data structures into large vectors.
If the embedding dimension is high enough, we can efficiently 
decode vectors back to the original data structure.  

These vectors can be used to compute on the underlying data structures,
in some cases without decoding.  BT embedding vectors are convenient building
blocks for explicit transformer models that compute on recursive data
structures.

For more info: [Read the paper](paper.pdf) or [check out the demo](Demo.ipynb).
