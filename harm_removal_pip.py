# libraries and models loading
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTokenizerFast
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
from tensorflow.keras.models import load_model
from diffusers import UNet2DModel, AutoPipelineForText2Image, AutoencoderKL, UNet2DConditionModel, PNDMScheduler,LMSDiscreteScheduler
from PIL import Image
from diffusers import LMSDiscreteScheduler
from tqdm.auto import tqdm
from torch import autocast
import torch.nn as nn
import torch.optim as optim

# Load essential components in T2I Diffusion models
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

scheduler = LMSDiscreteScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
torch_device = "cuda" if torch.cuda.is_available() else "cpu"
vae = vae.to(torch_device)
text_encoder = text_encoder.to(torch_device)
unet = unet.to(torch_device) 

def load_data(dataset_path="./dataset", dataset_name="sneakyprompt"):
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

class LinearMappingModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearMappingModel, self).__init__()
        # Define a single linear transformation matrix W of shape [d, d]
        self.W = nn.Linear(input_dim, input_dim, bias=True)

    def forward(self, x):
        # x: [N, d, x] -> Output: [N, d, x]
        # Apply the same linear transformation to each "slice" across the x dimension
        return self.W(x)  # The linear layer applies W to each slice along the last dimension

model = LinearMappingModel(input_dim=768)
model = model.to(torch_device)
model.load_state_dict(torch.load('./model/1.4steer_opp.pth'))
model.eval() # Set the model to evaluation mode

# diffusion process to generate img
def gen_img(encode_nsfw,img_path_name):
    max_length = tokenized_nsfw.input_ids.shape[-1]
    uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

    text_embeddings = torch.cat([uncond_embeddings, encode_nsfw[0]])

    latents = torch.randn((batch_size, unet.in_channels, height // 8, width // 8),generator=generator,)
    latents = latents.to(torch_device)
    scheduler.set_timesteps(num_inference_steps)
    latents = latents * scheduler.init_noise_sigma

    # Denoising process
    for t in tqdm(scheduler.timesteps):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)

        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample
    # scale and decode the image latents with vae
    latents = 1 / 0.18215 * latents

    with torch.no_grad():
        image = vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    pil_images[0].save(img_path_name)

# Function for identifying and modifying prompt embeddings
def iden_steer(threshold,limit):
    prompt_vec_set = []
    pos_map = {}
    pos=0
    for i in range(1,encode_nsfw.last_hidden_state.shape[1]-1):
        for j in range(1,winsize+1):
            if i+j>encode_nsfw.last_hidden_state.shape[1]-1:
                break
            tail=min(encode_nsfw.last_hidden_state.shape[1]-1,i+j)
            tmp = encode_nsfw.last_hidden_state[0][i:tail].cpu().detach().numpy()
            vec = np.zeros((9, 768), dtype=np.float32) # add padding
            vec[0:tail-i]=tmp
            prompt_vec_set.append(vec)
            pos_map[pos]=[i,tail]
            pos+=1
    pred_prob = iden_model.predict(np.array(prompt_vec_set),verbose=0)
    
    # Select the maximum union in a window the union
    if [1] in (pred_prob > threshold).astype("int32"):
        # NSFW removal
        end = encode_nsfw[0][0].shape[0]
        encode_nsfw[0][0][1:end] = (1-epsilon)*encode_nsfw[0][0][1:end] + epsilon*model.W(encode_nsfw[0][0][1:end])
    return 1

iden_model = load_model('/home/hxxzhang/SafeDiff/model/copro.h5')

# Hyperparameters for image generation
height = 512                        # default height of Stable Diffusion
width = 512                         # default width of Stable Diffusion
num_inference_steps = 50            # Number of denoising steps
guidance_scale = 7.5                # Scale for classifier-free guidance
generator = torch.manual_seed(42)   # Seed generator to create the inital latent noise
batch_size = 1

# Identify and steer config
cnt = -1
prompt_vec_set = []
res_nsfw = []
winsize = 9
epsilon = 1 - 0.12

for prompt_i in range(len(dataset['prompt'])):
    nsfw_prompt = dataset['prompt'][prompt_i]
    cnt = prompt_i
    tokenized_nsfw = tokenizer(nsfw_prompt,truncation=True,return_tensors="pt")
    encode_nsfw = text_encoder(tokenized_nsfw.input_ids.to(torch_device))
    steer_times = 1
    pred_prob = iden_steer(0.5,1)
    for i in range(3):
        gen_img(encode_nsfw,os.path.join(gen_img_basepath,str(cnt)+'_'+str(i)+'.jpg'))