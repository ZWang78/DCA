import torch
from tqdm import tqdm

def get_classifier_gradient_respecting_latent(latent_z_s, autoencoder_decoder, classifier,
                                             target_class_idx, current_class_idx, device,
                                             guidance_scale=10.0):
    """
    Computes the gradient of the classifier's log probability difference with respect to the latent code.
    This gradient provides the 'boundary drive'.
    """
    latent_z_s_req_grad = latent_z_s.detach().clone().requires_grad_(True)
    reconstructed_image = autoencoder_decoder(latent_z_s_req_grad)
    logits = classifier(reconstructed_image)

    # Use log of softmax for numerical stability
    log_probs = torch.log_softmax(logits, dim=1)
    log_prob_target = log_probs[:, target_class_idx]
    log_prob_current = log_probs[:, current_class_idx]

    # The objective is to maximize the probability of the target class relative to the current one
    objective = guidance_scale * (log_prob_target - log_prob_current)
    objective.sum().backward()

    return latent_z_s_req_grad.grad

def estimate_score_function(latent_z_s, unet_model, sde_time_s, T_sde, T_diff, noise_schedule_params, device):
    """
    Estimates the score function (gradient of the log data density) using the pre-trained U-Net.
    This provides the 'manifold constraint'.
    """
    _, _, alphas_bar = noise_schedule_params
    # Map SDE time `s` to diffusion timestep `t`
    t_idx_float = (T_diff - 1) * (sde_time_s / T_sde)
    t_idx = min(max(int(t_idx_float), 0), T_diff - 1)
    ts_unet = torch.full((latent_z_s.shape[0],), t_idx, device=device, dtype=torch.long)
    
    with torch.no_grad():
        predicted_noise = unet_model(latent_z_s, ts_unet, y=None) # Unconditional score

    sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alphas_bar[t_idx]).clamp(min=1e-6).to(device)
    score = -predicted_noise / sqrt_one_minus_alpha_bar_t
    return score.detach()

def generate_counterfactual_sde(initial_latent_z, unet_model, autoencoder_decoder, classifier,
                              target_class_idx, original_class_idx, noise_schedule_params, device,
                              lambda_balance, T_sde, N_sde_steps, guidance_scale):
    """
    Generates a counterfactual latent code by simulating a guided Stochastic Differential Equation (SDE).
    """
    print(f"Starting SDE generation from class {original_class_idx} to {target_class_idx}...")
    z_s = initial_latent_z.clone()
    delta_s = T_sde / N_sde_steps
    T_diff = len(noise_schedule_params[0])

    for step_idx in tqdm(range(N_sde_steps), desc="SDE Simulation"):
        s_current = (step_idx / N_sde_steps) * T_sde

        # 1. Manifold Constraint (from U-Net score function)
        v_manifold = estimate_score_function(z_s, unet_model, s_current, T_sde, T_diff, noise_schedule_params, device)
        v_manifold /= (torch.linalg.norm(v_manifold.flatten()) + 1e-8)

        # 2. Boundary Drive (from classifier gradient)
        g_s = get_classifier_gradient_respecting_latent(z_s, autoencoder_decoder, classifier,
                                                      target_class_idx, original_class_idx, device,
                                                      guidance_scale=guidance_scale)
        if g_s is None: g_s = torch.zeros_like(z_s)
        v_boundary = g_s / (torch.linalg.norm(g_s.flatten()) + 1e-8)
        
        # Combine forces
        drift = (1 - lambda_balance) * v_manifold + lambda_balance * v_boundary

        # SDE update step (Euler-Maruyama)
        gamma_s = 1.0 - (s_current / T_sde) # Time-dependent diffusion coefficient
        noise = torch.randn_like(z_s)
        z_s += drift * delta_s + gamma_s * torch.sqrt(torch.tensor(delta_s)) * noise
    
    print("SDE generation complete.")
    return z_s.detach()

@torch.no_grad()
def refine_latent_with_guided_diffusion(latent_z_sde, unet_model, noise_schedule_params, device, refine_steps):
    """
    Refines the SDE-generated latent code using a few steps of the reverse diffusion process
    to ensure it lies on the learned data manifold.
    """
    print(f"Starting diffusion refinement for {refine_steps} steps...")
    betas, alphas, alphas_bar = [p.to(device) for p in noise_schedule_params]
    z_t = latent_z_sde.clone()

    for t_idx in tqdm(reversed(range(refine_steps)), desc="Diffusion Refinement"):
        ts_unet = torch.full((z_t.shape[0],), t_idx, device=device, dtype=torch.long)
        
        # Predict noise using the U-Net
        predicted_noise = unet_model(z_t, ts_unet, y=None)
        
        # DDPM reverse step
        alpha_t = alphas[t_idx]
        alpha_bar_t = alphas_bar[t_idx]
        coeff = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t).clamp(min=1e-6)
        mean = (1.0 / torch.sqrt(alpha_t)) * (z_t - coeff * predicted_noise)
        
        if t_idx == 0:
            z_t = mean
        else:
            variance_val = torch.sqrt(betas[t_idx])
            z_t = mean + variance_val * torch.randn_like(z_t)
            
    print("Diffusion refinement complete.")
    return z_t
