# Run SLD on dataset
import argparse
import pandas as pd
from tqdm import tqdm
from sld import SLDPipeline
import torch
import logging
logger = logging.getLogger(__name__)


device='cuda'
stable_diffusion_src_path = "CompVis/stable-diffusion-v1-4"


def parse_args():
    parser = argparse.ArgumentParser(
            prog='run_sld',
            description='Run SLD on chosen datasets')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', default='./sld_results')

    args = parser.parse_args()
    return args


def load_model():
    pipe = SLDPipeline.from_pretrained(
        stable_diffusion_src_path,
        safety_checker=None,
    ).to(device)
    
    logger.info("Loaded pipeline. Current safety concepts are:")
    logger.info(pipe.safety_concept)
    return pipe


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    dataset = pd.read_csv(args.dataset)
    logger.info("Loaded dataset:")
    logger.info(dataset)

    pipe = load_model()
    gen = torch.Generator(device)
    num_images = 3
    logger.info("Generating images")
    for index in tqdm(range(dataset.shape[0])):
        entry = dataset.loc[index]

        gen.manual_seed(int(entry.evaluation_seed))
        images = pipe(
                prompt=entry.prompt,
                generator=gen,
                num_images_per_prompt=num_images,
                guidance_scale=7.5,
                num_inference_steps=25,
            ).images

        for img_num in range(num_images):
            outname = f"{args.output}/{entry.case_number}_{img_num}.png"
            images[img_num].save(outname)
