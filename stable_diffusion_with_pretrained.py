from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, AutoencoderKL
from diffusers.utils import load_image, make_image_grid
from PIL import Image
import cv2
import numpy as np
import torch
import os
from datetime import datetime
import itertools


def prepare_image_for_qr_control(image_path, save_path=None):
    original = Image.open(image_path).convert("RGB")
    img_array = np.array(original)
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    _, binary = cv2.threshold(enhanced, 150, 255, cv2.THRESH_BINARY)
    
    binary_image = Image.fromarray(binary).resize((512, 512))
    
    rgb_image = Image.new("RGB", binary_image.size)
    rgb_image.paste(binary_image)

    if save_path is not None:
        rgb_image.save(save_path)
        print(f"Processed image saved to: {save_path}")
    
    return rgb_image


def process_image(image_path, output_dir, prompts, control_scales, num_images_per_set=3, guidance_scale=12, num_inference_steps=40):
    """
    Process an image with multiple prompts and control scales using QR ControlNet.
    
    Args:
        image_path: Path to the input image
        output_dir: Directory to save outputs
        prompts: List of (prompt, negative_prompt) tuples
        control_scales: List of control conditioning scale values
        num_images_per_set: Number of images to generate for each parameter combination
        guidance_scale: Fixed guidance scale value
        num_inference_steps: Number of inference steps
    """
    # Load and prepare the original image with enhanced preprocessing
    control_image = prepare_image_for_qr_control(image_path, "Gan_project/QR_try/edited_guide1.png")
    
    # Save the preprocessed control image for inspection
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    preprocessed_dir = os.path.join(output_dir, "preprocessed")
    os.makedirs(preprocessed_dir, exist_ok=True)
    control_image.save(os.path.join(preprocessed_dir, f"{base_filename}_preprocessed.png"))
    
    # Create a subdirectory for this specific digit
    digit_output_dir = os.path.join(output_dir, f"digit_{base_filename}")
    os.makedirs(digit_output_dir, exist_ok=True)

    # Create combination of prompts and control scales
    param_combinations = list(itertools.product(
        enumerate(prompts),
        enumerate(control_scales)
    ))

    total_combinations = len(param_combinations)
    total_images = total_combinations * num_images_per_set
    print(f"\nProcessing {total_combinations} combinations for digit {base_filename}")
    print(f"Will generate {total_images} total images ({num_images_per_set} per combination)")

    for (prompt_idx, (prompt, neg_prompt)), (scale_idx, control_scale) in param_combinations:
        print(f"\nGenerating with parameters:")
        print(f"Prompt: {prompt}")
        print(f"Control Scale: {control_scale}")
        print(f"Guidance Scale: {guidance_scale}")
        
        # Generate multiple images for this parameter set
        for img_num in range(num_images_per_set):
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                generator = torch.Generator(device="cuda").manual_seed(42 + img_num)
                
                generated_image = pipe(
                    prompt,
                    negative_prompt=neg_prompt,
                    image=control_image,
                    controlnet_conditioning_scale=control_scale,
                    num_inference_steps=num_inference_steps + 20,  # More steps for better detail
                    generator=generator,
                    guidance_scale=guidance_scale
                ).images[0]

                params_str = f"p{prompt_idx+1}_cs{control_scale}_n{img_num+1}"
                single_image_filename = os.path.join(
                    digit_output_dir, 
                    f"{base_filename}_{params_str}_{timestamp}.png"
                )

                generated_image.save(single_image_filename)
                print(f"Successfully saved: {os.path.basename(single_image_filename)} ({img_num+1}/{num_images_per_set})")
                
            except Exception as e:
                print(f"Error with parameters {params_str}, image {img_num+1}: {str(e)}")
                continue


def main():
    global pipe
    output_dir = "your output path"
    os.makedirs(output_dir, exist_ok=True)

    controlnet = ControlNetModel.from_pretrained(
        "monster-labs/control_v1p_sd15_qrcode_monster",
        torch_dtype=torch.float16,
        use_safetensors=True
    )

    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse", 
        torch_dtype=torch.float16, 
        use_safetensors=True
    )

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "SG161222/Realistic_Vision_V5.1_noVAE",
        controlnet=controlnet,
        vae=vae,
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    pipe.enable_model_cpu_offload()

    prompts = [
        ("a feild of tall sunflowers", "people, low quality, bad quality, sketches")]

    control_scales = [0.7, 0.8]

    # Define input images
    image_paths = ["input paths of images for controlnet"]

    for image_path in image_paths:
        print(f"\nProcessing {os.path.basename(image_path)}...")
        process_image(
            image_path=image_path,
            output_dir=output_dir,
            prompts=prompts,
            control_scales=control_scales,
            num_images_per_set=2,
            guidance_scale=10
        )

    print("\nImage generation complete!")

if __name__ == "__main__":
    main()
