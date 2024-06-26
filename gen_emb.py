import torch
from transformers import AutoTokenizer, OPTForCausalLM
import pandas as pd
import numpy as np
from typing import Dict, List
import json
from pathlib import Path
import logging
from tqdm import tqdm
import argparse,os
import csv
from diffusers import UNet2DModel, AutoPipelineForText2Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

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

def process_batch(batch_prompts: List[str], model, tokenizer, layers_to_use: list, remove_period: bool):
    """
    Processes a batch of data and returns the embeddings for each statement.
    """
    if remove_period:
        batch_prompts = [prompt.rstrip(". ") for prompt in batch_prompts]
    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True) 

    # Use the attention mask to find the index of the last real token for each sequence
    seq_lengths = inputs.attention_mask.sum(dim=1) - 1  # Subtract 1 to get the index

    batch_embeddings = {}
    for layer in layers_to_use:
        hidden_states = outputs.hidden_states[layer]

        # Gather the hidden state at the last real token for each sequence
        last_hidden_states = hidden_states[range(hidden_states.size(0)), seq_lengths, :]
        batch_embeddings[layer] = [embedding.detach().cpu().numpy().tolist() for embedding in last_hidden_states]

    return batch_embeddings


def get_hook(name, layer_outputs):
    def hook(module, input, output):
        layer_outputs[name] = output
    return hook

def main():
    parser = argparse.ArgumentParser(description="Generate new csv with embeddings.")
    parser.add_argument("--dataset_path", default="./dataset")
    parser.add_argument("--dataset_names", default='test')
    args = parser.parse_args()
    dataset = load_data(args.dataset_path,args.dataset_names)
    df=pd.DataFrame()

    for i in range(len(dataset['prompt'])):
        input_text = dataset['prompt'][i]
        layer_outputs = {}
        pipeline = AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16", requires_safety_checker=True).to("cuda")
        # Register hooks to the layers
        for name, module in pipeline.text_encoder.named_modules():
            # if name=='text_model' or name=='text_model.embeddings':
            module.register_forward_hook(get_hook(f"text_encoder_{name}", layer_outputs))
            
        image = pipeline(input_text).images[0]
        # tmp = layer_outputs['text_encoder_text_model']['pooler_output'].cpu().numpy()[0]
        print(input_text,layer_outputs)
        # df = pd.concat([df, pd.DataFrame([tmp])], ignore_index=True)
    df.to_csv('vec.csv',index=False)


    embeddings = df.to_numpy()
    # Plot the t-SNE results
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_tsne = tsne.fit_transform(embeddings)
    # Create a DataFrame with the t-SNE results
    tsne_df = pd.DataFrame(embeddings_tsne, columns=['Dim1', 'Dim2'])
    tsne_df['label'] = df['label']
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=tsne_df, x='Dim1', y='Dim2', hue='label', palette='viridis', s=50, alpha=0.7)
    plt.title('t-SNE of Embeddings')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend(title='Label', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig("sne.png")

    # Perform PCA to reduce the dimensionality to 2
    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(embeddings)
    # Create a DataFrame with the PCA results
    pca_df = pd.DataFrame(embeddings_pca, columns=['PC1', 'PC2'])
    pca_df['label'] = df['label']
    # Plot the PCA results
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='label', palette='viridis', s=50, alpha=0.7)
    plt.title('PCA of Embeddings')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Label', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig("pca.png")

if __name__ == "__main__":
    main()