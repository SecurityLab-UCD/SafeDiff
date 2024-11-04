# SafeDiff
Defense prompt jailbreak to diffusion models

## Dataset Setup
The datasets used in our experiments are located in the `dataset/` directory.

- `./dataset/nsfw`: Contains a set of jailbreak prompt datasets.
- `./dataset/sfw`: Contains a set of safe prompt datasets.

## Model Setup
The models used in our experiments are stored in the `model/` directory. You can also create and debug your own models using the following scripts:

- `./harm_iden.ipynb`: Used to develop and train the identification model.
- `./harm_removal.ipynb`: Used to build and fine-tune the steered model.

## Generate Images
To generate images, use the following scripts:

- `./SD_gen_img.py`: Implements the Stable Diffusion 1.4 pipeline for image generation.
- `./SLD_gen_img.py`: Implements the SLD pipeline for image generation. Refer to [Safe Latent Diffusion (SLD)](https://github.com/ml-research/safe-latent-diffusion).
- `./harm_removal_pip.py`: Implements the SteerDiff pipeline for image generation.

## Evaluation
To evaluate the results of the generated images, we apply both **NudeNet** and **Q16**.

- `./nudenet_iden.ipynb`: Provides detailed experimental results using NudeNet.
- `./data`: Contains visualization data for analysis.

## Acknowledgement
This project utilizes and is inspired by the following resources:

- **NudeNet**: [https://github.com/notAI-tech/NudeNet](https://github.com/notAI-tech/NudeNet)
- **ESD**: [https://erasing.baulab.info/](https://erasing.baulab.info/)
- **SLD**: [https://github.com/ml-research/safe-latent-diffusion](https://github.com/ml-research/safe-latent-diffusion)
- **Q16**: [https://github.com/ml-research/Q16](https://github.com/ml-research/Q16)

## Model architecture

### Stable diffusion v1.4
#### text_encoder
```
CLIPTextModel(
  (text_model): CLIPTextTransformer(
    (embeddings): CLIPTextEmbeddings(
      (token_embedding): Embedding(49408, 768)
      (position_embedding): Embedding(77, 768)
    )
    (encoder): CLIPEncoder(
      (layers): ModuleList(
        (0-11): 12 x CLIPEncoderLayer(
          (self_attn): CLIPAttention(
            (k_proj): Linear(in_features=768, out_features=768, bias=True)
            (v_proj): Linear(in_features=768, out_features=768, bias=True)
            (q_proj): Linear(in_features=768, out_features=768, bias=True)
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
          )
          (layer_norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (mlp): CLIPMLP(
            (activation_fn): QuickGELUActivation()
            (fc1): Linear(in_features=768, out_features=3072, bias=True)
            (fc2): Linear(in_features=3072, out_features=768, bias=True)
          )
          (layer_norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
      )
    )
    (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  )
)
```