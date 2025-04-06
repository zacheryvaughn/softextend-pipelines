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
image = Image.open("outpaint_images/stirred_output.png")
mask = Image.open("outpaint_images/MASK_1.png")

# INPUT PROMPTS
prompt = "flat straight ocean horizon and beach shoreline during sunset"
negative_prompt = ""

# RUN PIPELINE
output = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=image,

    # If you do not provide the "extend_sides" parameter or it equals 0 on all sides, then "noise_fill_image" and "auto_mask" will be False.
    # It will expect you to provide a mask image and an extended and painted image.
    # Using an image with manually painted extension areas and a custom mask can produce better results but is obviously less convenient.
    extend_sides=[64, 64, 64, 64], # Top, Right, Bottom, Left (must be 8px divisible, the pipeline will round down if not)
    # auto_mask and noise_fill_image are true if extend_sides

    num_inference_steps=32,
    guidance_scale=6,
    strength=0.9,
).images[0]

# SAVE OUTPUT
output.save("output.png")
print("Done!")