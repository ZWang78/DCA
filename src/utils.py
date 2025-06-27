import torch
import math

def get_noise_schedule(timesteps, schedule_type='linear'):
    """
    Generates a noise schedule (betas, alphas, and cumulative alphas).
    """
    if schedule_type == 'linear':
        betas = torch.linspace(0.0001, 0.02, timesteps)
    elif schedule_type == 'cosine':
        s = 0.008
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_bar = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_bar = alphas_bar / alphas_bar[0]
        betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
        betas = torch.clamp(betas, 0.0001, 0.9999)
    else:
        raise NotImplementedError(f"Schedule '{schedule_type}' is not implemented.")
    
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_bar

@torch.no_grad()
def diagnose_counterfactual_generation(original_image, cf_image, classifier, autoencoder):
    """
    Prints diagnostic information comparing the original and counterfactual images.
    """
    # Get classifier predictions
    orig_logits = classifier(original_image)
    cf_logits = classifier(cf_image)
    orig_probs = torch.softmax(orig_logits, dim=1).squeeze().cpu().numpy()
    cf_probs = torch.softmax(cf_logits, dim=1).squeeze().cpu().numpy()
    
    # Analyze image and latent differences
    pixel_diff = torch.abs(original_image - cf_image).mean().item()
    orig_mean, _ = torch.chunk(autoencoder.encoder(original_image), 2, dim=1)
    cf_mean, _ = torch.chunk(autoencoder.encoder(cf_image), 2, dim=1)
    latent_diff = torch.abs(orig_mean - cf_mean).mean().item()
    
    print("\n--- Counterfactual Generation Diagnosis ---")
    print(f"Original Image Probs:  {orig_probs}")
    print(f"Counterfact. Image Probs: {cf_probs}")
    print("-" * 20)
    print(f"Prediction Change:      {orig_logits.argmax().item()} -> {cf_logits.argmax().item()}")
    print(f"Mean Pixel Difference:  {pixel_diff:.6f}")
    print(f"Mean Latent Difference: {latent_diff:.6f}")
    print("-----------------------------------------\n")
