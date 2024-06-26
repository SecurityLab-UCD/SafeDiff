from diffusers import AutoPipelineForText2Image
import pandas as pd
import argparse
import os
import torch

def load_data(dataset_path="./dataset", dataset_name="i2p_benchmark"):
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
    parser.add_argument("--dataset_path", default="./dataset")
    parser.add_argument("--dataset_names", default='i2p_benchmark')
    args = parser.parse_args()
    dataset = load_data(args.dataset_path,args.dataset_names)
    pipeline = AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16", requires_safety_checker=True).to("cuda")
    for i in range(len(dataset['prompt'])):
        image = pipeline(dataset['prompt'][i]).images[0]
        image.save(os.path.join("./gen_img",f"{i}.jpg"))

if __name__ == "__main__":
    main()