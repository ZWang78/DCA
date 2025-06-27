import torch

# --- Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Data and Image Parameters ---
IMG_MEAN = [0.6283, 0.6283, 0.6283]
IMG_STD = [0.1531, 0.1531, 0.1531]
IMAGE_SIZE = 224
IMAGE_CHANNELS = 3

# --- Model Hyperparameters ---
# Autoencoder
AE_CHANNELS = [32, 64, 128, 256]
AE_NUM_RES_BLOCKS = 1
LATENT_DIM = 8 # Latent channel dimension
LATENT_SPATIAL_SIZE = 28 # H_latent, W_latent

# U-Net
EMB_DIM = 128
UNET_CHANNELS = [EMB_DIM, EMB_DIM * 2, EMB_DIM * 4]
UNET_NUM_RES_BLOCKS = 2
ATTENTION_RESOLUTIONS = [LATENT_SPATIAL_SIZE // 2, LATENT_SPATIAL_SIZE]
UNET_DROPOUT = 0.1

# Diffusion
T_DIFF = 200
NOISE_SCHEDULE = 'linear'

# --- Classifier Configuration ---
CLASSIFIER_CONFIG = {
    "MODEL_NAME": "vit_base_patch16_224",
    "PRETRAINED": True,
    "NUM_CLASSES": 2, # Binary classification for KL2 vs KL3
    "OUTPUT_DIR": "./vit_classifier_kl23/",
    "WEIGHTS_NAME": "best_model_20250506.pth"
}

# --- SDE and Generation Parameters ---
SDE_LAMBDA_BALANCE = 1.0       # Balance between manifold and boundary drive (1.0 = full boundary drive)
SDE_T = 10.0                   # Total simulation time
SDE_N_STEPS = 1000             # Number of simulation steps
SDE_GUIDANCE_SCALE = 50.0      # Strength of the classifier gradient guidance

# --- Refinement Parameters ---
REFINE_STEPS = 50              # Number of reverse diffusion steps for refinement
REFINE_GUIDANCE_SCALE = 3.0    # Strength of classifier guidance during refinement
