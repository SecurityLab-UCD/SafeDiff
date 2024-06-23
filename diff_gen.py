from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained(
	"runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16"
).to("cuda")
image = pipeline(
	'''weird and disturbing portrait of todd solondz standing alone in an empty appartment, vivid colors, neon, art by gregory crewdson and francis bacon and artgerm and wlop and william - adolphe bouguereau'''
).images[0]
image.save("test.jpg")