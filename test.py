from PIL import Image
from prep import stir_outpaint_colors_pipeline

image = Image.open("image.png")
extend_sides=[0, 384, 0, 384]

outpainted, mask = stir_outpaint_colors_pipeline(
    image=image,
    extend_sides=extend_sides,
).images[0]

outpainted.save("outpainted.png")
mask.save("mask.png")