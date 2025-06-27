import torch
import matplotlib.pyplot as plt
import numpy as np
import timm
import os

from configs.config import *
from src.data_loader import SingleImageDataset, TransformedDatasetWrapper, train_transforms
from src.models.autoencoder import Autoencoder
from src.models.unet import UNetModel
from src.sde_generator import generate_counterfactual_sde, refine_latent_with_diffusion
from src.utils import get_noise_schedule, diagnose_generation

def main():
    """Main function to generate and visualize a counterfactual image."""
    print(f"Using device: {DEVICE}")

    # --- 1. Define Image and Target ---
    # IMPORTANT: Update this path to your source image
    source_image_path = r"C:\Users\wz\Desktop\CF_dataset_sub_KL23\2\2_9009927R.png"
    original_kl_grade = 2
    target_map = {2: 0, 3: 1} # KL Grade to classifier index

    if not os.path.exists(source_image_path):
        print(f"Error: Source image not found at {source_image_path}")
        print("Please update the 'source_image_path' in generate_counterfactual.py")
        return

    original_class_idx = target_map[original_kl_grade]
    target_class_idx = 1 - original_class_idx # Flip for binary classification

    # --- 2. Load Models ---
    print("Loading pre-trained models...")
    autoencoder = Autoencoder(IMAGE_CHANNELS, AE_CHANNELS, AE_NUM_RES_BLOCKS, LATENT_DIM).to(DEVICE)
    autoencoder.load_state_dict(torch.load('AE_conter_RGB_KL23.pth', map_location=DEVICE))
    autoencoder.eval()

    unet_model = UNetModel(LATENT_DIM, UNET_CHANNELS, LATENT_DIM, UNET_NUM_RES_BLOCKS, 
                           ATTENTION_RESOLUTIONS, UNET_DROPOUT, CLASSIFIER_CONFIG["NUM_CLASSES"], EMB_DIM).to(DEVICE)
    unet_model.load_state_dict(torch.load('diff_KL23.pth', map_location=DEVICE))
    unet_model.eval()

    classifier = timm.create_model(CLASSIFIER_CONFIG["MODEL_NAME"], pretrained=False, num_classes=CLASSIFIER_CONFIG["NUM_CLASSES"]).to(DEVICE)
    classifier.load_state_dict(torch.load(CLASSIFIER_CONFIG["WEIGHTS_PATH"], map_location=DEVICE))
    classifier.eval()
    print("All models loaded.")

    # --- 3. Prepare Data ---
    dataset = SingleImageDataset(source_image_path, original_kl_grade)
    wrapped_data = TransformedDatasetWrapper(dataset, train_transforms, target_map)
    image_tensor, _ = wrapped_data[0]
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    # --- 4. Run Generation Pipeline ---
    schedule = get_noise_schedule(T_DIFF, NOISE_SCHEDULE)
    with torch.no_grad():
        _, initial_latent, _ = autoencoder(image_tensor)

    latent_sde = generate_counterfactual_sde(initial_latent, unet_model, autoencoder.decoder, classifier,
                                             target_class_idx, original_class_idx, schedule, DEVICE,
                                             SDE_LAMBDA_BALANCE, SDE_T, SDE_N_STEPS, SDE_GUIDANCE_SCALE)
    
    refined_latent = refine_latent_with_diffusion(latent_sde, unet_model, schedule, DEVICE, REFINE_STEPS)
    
    with torch.no_grad():
        counterfactual_image = autoencoder.decoder(refined_latent)

    # --- 5. Display and Diagnose ---
    diagnose_generation(image_tensor, counterfactual_image, classifier, autoencoder)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy())
    axes[0].set_title(f"Original (KL Grade {original_kl_grade})")
    axes[0].axis('off')

    axes[1].imshow(counterfactual_image.squeeze(0).cpu().permute(1, 2, 0).numpy())
    axes[1].set_title(f"Counterfactual (Target KL Grade {list(target_map.keys())[target_class_idx]})")
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig("counterfactual_result.png")
    plt.show()

if __name__ == "__main__":
    main()
