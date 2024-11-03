from diffusers import AutoPipelineForText2Image, DiffusionPipeline, StableDiffusionPipelineSafe
from diffusers.pipelines.stable_diffusion_safe import SafetyConfig
import pandas as pd
import argparse
import os
import torch

def load_data(dataset_path="./dataset/nsfw", dataset_name="i2p_benchmark"):
    dataset_file = os.path.join(dataset_path, f"{dataset_name}.csv")
    try:
        df = pd.read_csv(dataset_file)
    except FileNotFoundError as e:
        print(f"Dataset file {dataset_file} not found: {str(e)}")
        return None
    except pd.errors.ParserError as e:
        print(f"Error parsing the CSV file {dataset_file}: {str(e)}")
        return None
    except pd.errors.EmptyDataError as e:
        print(f"No data in CSV file {dataset_file}: {str(e)}")
        return None
    if 'embeddings' not in df.columns:
        df['embeddings'] = pd.Series(dtype='object')
    return df

def main():
    parser = argparse.ArgumentParser(description="Generate new csv with embeddings.")
    parser.add_argument("--dataset_path", default="./dataset/nsfw")
    parser.add_argument("--dataset_names", default='i2p_benchmark')
    args = parser.parse_args()
    dataset = load_data(args.dataset_path,args.dataset_names)
    # Set parameters
    num_inference_steps = 25            # Number of denoising steps
    guidance_scale = 7.5                # Scale for classifier-free guidance
    # batch_size = 1                      # Batch size
    access_token = "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"

    # pipeline = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16, variant="fp16").to("cuda")
    pipeline = StableDiffusionPipelineSafe.from_pretrained("AIML-TUDA/stable-diffusion-safe",token=access_token,).to("cuda")
    pipeline.safety_checker = None
    
    for i in range(len(dataset['prompt'])):
        for j in range(3):
            image = pipeline(prompt=dataset['prompt'][i],num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            # batch_size=batch_size,
            **SafetyConfig.MAX
            ).images[0]
            image.save(os.path.join("/home/hxxzhang/SafeDiff/gen_img/sld_MAX_step25",f"{i}_{j}.jpg"))

if __name__ == "__main__":
    main()