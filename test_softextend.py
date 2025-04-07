import torch
from PIL import Image
from pipeline_stable_diffusion_xl_softextend import StableDiffusionXLSoftExtendPipeline

# SET DEVICE & PRECISION
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
torch_dtype = torch.float16 if device in ["cuda", "mps"] else torch.float32

# LOAD PIPELINE
print("Loading pipeline...")
pipeline = StableDiffusionXLSoftExtendPipeline.from_single_file(
    "StableDiffusionXL.safetensors", # Add your model here.
    torch_dtype=torch_dtype,
    use_safetensors=True,
).to(device)

# PIL OPEN IMAGES
image = Image.open("image.png")

# INPUT PROMPTS
prompt = "hires photo of beautiful mountain scenery with wildflowers in the valley"
negative_prompt = "lowres, bad quality, frame, border"

# RUN PIPELINE
output = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=image,

    # If you do not provide the "extend_sides" parameter or it equals 0 on all sides, then
    # it will expect you to provide a mask and an extended+painted image.
    # Using an image with manually painted extension areas and a custom mask can produce better results but is less convenient.
    extend_sides=[0, 256, 0, 256], # [Top, Right, Bottom, Left] (must be 8px divisible, the pipeline will round down if not)

    num_inference_steps=50,
    guidance_scale=6,
    strength=0.85,
).images[0]

# SAVE OUTPUT
output.save("output.png")
print("Done!")