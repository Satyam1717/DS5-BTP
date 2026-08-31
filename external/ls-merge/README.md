# LS-Merge: Merging Language Models in Latent Space
> Merging LLM in weights latent space.

example of vae training

 CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes=1 --nproc_per_node=4 train_tf_vae.py