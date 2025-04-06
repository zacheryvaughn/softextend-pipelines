import cv2
import numpy as np
import random
from noise import pnoise2
from PIL import Image, ImageFilter

image = Image.open("image.png")
extend_sides=[0, 384, 0, 384]

def generate_displacement_map(shape, scale=100, octaves=4, seed=None, strength=10):
    height, width = shape
    if seed is None:
        seed = random.randint(0, 1_000_000)
    rng = random.Random(seed)
    offset_x = rng.uniform(-1000, 1000)
    offset_y = rng.uniform(-1000, 1000)

    # Create coordinate grids
    y_indices, x_indices = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    nx = (x_indices + offset_x) / scale
    ny = (y_indices + offset_y) / scale

    # pnoise2 does not support vectorized input, so we wrap it with np.vectorize.
    v_noise = np.vectorize(lambda a, b: pnoise2(a, b, octaves=octaves))
    angles = v_noise(nx, ny) * 2 * np.pi

    dx = np.cos(angles) * strength
    dy = np.sin(angles) * strength
    return dx.astype(np.float32), dy.astype(np.float32)

def apply_displacement(image, dx, dy):
    height, width = image.shape[:2]
    # Generate coordinate grids with 'xy' indexing for cv2.remap.
    map_x, map_y = np.meshgrid(np.arange(width), np.arange(height), indexing='xy')
    map_x = (map_x + dx).astype(np.float32)
    map_y = (map_y + dy).astype(np.float32)
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def fbm(x, y, scale, octaves, lacunarity, gain):
    total = 0.0
    amplitude = 1.0
    frequency = 1.0
    for _ in range(octaves):
        total += amplitude * pnoise2(x * frequency / scale, y * frequency / scale)
        amplitude *= gain
        frequency *= lacunarity
    return total

def pattern(x, y, scale, octaves, lacunarity, gain):
    q0 = fbm(x, y, scale, octaves, lacunarity, gain)
    q1 = fbm(x + 5.2, y + 1.3, scale, octaves, lacunarity, gain)
    return fbm(x + 80.0 * q0, y + 80.0 * q1, scale, octaves, lacunarity, gain)

def generate_pattern_noise(size=(512, 512), scale=80, octaves=5, lacunarity=2.0, gain=0.5,
                           saturation=1.5, brightness=1.0, seed=None):
    width, height = size
    rng = random.Random(seed)
    offset_x = rng.uniform(-1000, 1000)
    offset_y = rng.uniform(-1000, 1000)

    # Create coordinate grids
    y_indices, x_indices = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    x = x_indices + offset_x
    y = y_indices + offset_y

    # Use vectorization for the pattern computation.
    v_pattern = np.vectorize(lambda a, b: pattern(a, b, scale, octaves, lacunarity, gain))
    r_vals = v_pattern(x, y)
    g_vals = v_pattern(x + 100, y + 100)
    b_vals = v_pattern(x + 200, y + 200)

    # Normalize from [-1,1] to [0,1]
    r = (r_vals + 1) / 2
    g = (g_vals + 1) / 2
    b = (b_vals + 1) / 2

    # Adjust saturation based on the average
    avg = (r + g + b) / 3.0
    r = np.clip(avg + (r - avg) * saturation, 0, 1) * brightness
    g = np.clip(avg + (g - avg) * saturation, 0, 1) * brightness
    b = np.clip(avg + (b - avg) * saturation, 0, 1) * brightness

    # Stack channels together and convert to image.
    img = np.stack([r, g, b], axis=-1)
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    image = Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius=2))
    return image

def stir_outpaint_colors(input_path, output_path, extend_sides=[64, 64, 64, 64]):
    # === Hardcoded parameters ===
    scale = 100
    octaves = 4
    strength = 12
    seed = random.randint(0, 1_000_000)
    overlay_opacity = 0.5
    auto_mask = True
    mask_output_path = "mask_output.png"
    mask_overlap = 60
    mask_blur = 120
    noise_overlap = 40
    noise_blur = 80

    top, right, bottom, left = extend_sides

    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError("Image not found.")

    h, w = image.shape[:2]
    extended = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REPLICATE)
    full_h, full_w = extended.shape[:2]

    # === Create mask ===
    mask = np.zeros((full_h, full_w), dtype=np.float32)
    if top > 0:
        mask[:top + mask_overlap, :] = 1.0
    if bottom > 0:
        mask[-(bottom + mask_overlap):, :] = 1.0
    if left > 0:
        mask[:, :left + mask_overlap] = 1.0
    if right > 0:
        mask[:, -(right + mask_overlap):] = 1.0

    # Ensure the kernel size is odd for GaussianBlur.
    fade_kernel = mask_blur if mask_blur % 2 == 1 else mask_blur + 1
    mask_blurred = cv2.GaussianBlur(mask, (fade_kernel, fade_kernel), mask_blur)

    if auto_mask:
        mask_image = (mask_blurred * 255).astype(np.uint8)
        cv2.imwrite(mask_output_path, mask_image)
        print(f"✅ Saved black and white mask to {mask_output_path}")

    dx, dy = generate_displacement_map(mask.shape, scale, octaves, seed, strength)
    warped = apply_displacement(extended, dx, dy)

    mask_3ch = np.stack([mask_blurred] * 3, axis=2)
    final = (warped * mask_3ch + extended * (1 - mask_3ch)).astype(np.uint8)

    # === Overlay noise ===
    pattern_img = generate_pattern_noise((full_w, full_h), seed=seed)
    pattern_np = np.array(pattern_img)
    pattern_np = cv2.cvtColor(pattern_np, cv2.COLOR_RGB2BGR)

    overlay_mask = np.zeros((full_h, full_w), dtype=np.float32)
    if top > 0:
        overlay_mask[:top + noise_overlap, :] = 1.0
    if bottom > 0:
        overlay_mask[-(bottom + noise_overlap):, :] = 1.0
    if left > 0:
        overlay_mask[:, :left + noise_overlap] = 1.0
    if right > 0:
        overlay_mask[:, -(right + noise_overlap):] = 1.0

    overlay_kernel = noise_blur if noise_blur % 2 == 1 else noise_blur + 1
    overlay_mask = cv2.GaussianBlur(overlay_mask, (overlay_kernel, overlay_kernel), noise_blur)
    overlay_mask_3ch = np.stack([overlay_mask] * 3, axis=2)

    alpha = overlay_mask_3ch * overlay_opacity
    final = (final * (1 - alpha) + pattern_np * alpha).astype(np.uint8)

    cv2.imwrite(output_path, final)
    print(f"✅ Saved stirred image with seed {seed} to {output_path}")

    return image, mask

# Run the function with the specified parameters.
stir_outpaint_colors()
