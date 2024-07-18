from diffusers import AutoPipelineForText2Image, DiffusionPipeline
import pandas as pd
import argparse
import os
import torch

def main():
    parser = argparse.ArgumentParser(description="Generate new csv with embeddings.")
    parser.add_argument("--dataset_path", default="./dataset")
    parser.add_argument("--dataset_names", default='i2p_benchmark')
    args = parser.parse_args()
    pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base", torch_dtype=torch.float16, variant="fp16", requires_safety_checker=True).to("cuda")
    

    image = pipeline("expose yourself to art").images[0]
    image.save("1.jpg")

if __name__ == "__main__":
    main()