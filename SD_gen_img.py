from diffusers import AutoPipelineForText2Image, DiffusionPipeline
import pandas as pd
import argparse
import os
import torch

def main():
    # parser = argparse.ArgumentParser(description="Generate new csv with embeddings.")
    # parser.add_argument("--dataset_path", default="./dataset")
    # parser.add_argument("--dataset_names", default='i2p_benchmark')
    # args = parser.parse_args()
    num_inference_steps = 50            # Number of denoising steps
    guidance_scale = 7.5
    pipeline = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16,variant="fp16",num_inference_steps=num_inference_steps,guidance_scale=guidance_scale,requires_safety_checker=False).to("cuda")
    pipeline.safety_checker = None
    a = """A photo of Emma Watson."""
    for i in range(4):
        image = pipeline(a).images[0]
        image.save('Watson_'+str(i)+'_ori.jpg')
    

if __name__ == "__main__":
    main()