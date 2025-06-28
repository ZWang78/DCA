# Diffusion-based Counterfactual Augmentation for Knee Osteoarthritis Grading

[](https://arxiv.org/abs/2402.04868)
[](https://opensource.org/licenses/MIT)
[](https://pytorch.org/)

This repository contains the official implementation of **Diffusion-based Counterfactual Augmentation (DCA)**, a novel framework to improve the robustness and interpretability of models for Knee Osteoarthritis (KOA) grading, as presented in our paper.

-----

### 💡 Framework Overview

The DCA framework generates high-fidelity counterfactuals by steering data points across a classifier's decision boundary while remaining on the data manifold. These targeted examples are then used in a self-corrective loop to enhance the classifier's performance and robustness.

> *The process starts with an original data point **x** and generates a trajectory towards a target class to produce counterfactuals **x'**. This path is guided by a **boundary drive** from a frozen classifier and a **manifold constraint** from the learned data distribution to maintain realism. The generated data is used in a **self-corrective learning** loop to improve the model.*

### ✨ Key Features

  * **Targeted Counterfactual Generation**: We introduce a gradient-informed framework that explicitly targets classifier decision boundaries in the latent space. It generates meaningful radiographs reflecting subtle pathological transitions between adjacent KOA grades by balancing a classifier-driven boundary force with a learned manifold constraint.
  * **Self-Corrective Learning**: A pair of structurally identical classifiers (a frozen reference and a trainable counterpart) work together in a self-corrective loop. The static classifier guides counterfactual generation by identifying uncertain regions, while the learnable classifier is trained on these samples to resolve ambiguities and improve robustness.
  * **Comprehensive Validation**: We conduct extensive experiments on two large-scale KOA cohorts (OAI and MOST), evaluating the framework across multiple classification architectures. The results demonstrate consistent improvements in accuracy, generalizability, and interpretability.
  * **Enhanced Interpretability**: The framework offers intuitive visualizations of minimal anatomical changes and introduces `T_SDE_min` as a quantitative metric to probe the latent disease manifold, revealing a structure that aligns with the clinical understanding of KOA progression.

-----

### 🚀 Getting Started

#### Installation

  **Clone the repository:**


    git clone https://github.com/ZWang78/DCA.git
    cd DCA


#### Usage

1.  **Prepare Data & Models**:

      * Organize your dataset into folders named by their Kellgren-Lawrence (KL) grade.
      * Download the pre-trained weights for the Autoencoder, U-Net, and the reference classifier.
      * Update the paths in the main script (`generate_counterfactual.py`) and configuration file (`configs/config.py`).

2.  **Run Counterfactual Generation**:
    Execute the main script to generate a counterfactual image from a source image.

    ```bash
    python generate_counterfactual.py
    ```

    The script will display the original and generated images and save the result.

-----

### 🔧 Using a Custom Classifier

A core strength of the DCA framework is its modularity. You can **easily substitute the classifier** used for guidance and self-corrective learning with any architecture of your choice. Our paper validates this by showing performance gains across various models, including VGG, EfficientNet, and ViT.

To use your own classifier, follow these steps:

1.  **Ensure Compatibility**: Your classifier must be a `torch.nn.Module` that accepts a batch of images and outputs raw logits.

2.  **Modify the Loading Script**: Open `generate_counterfactual.py` (or your training script) and navigate to the model loading section. Replace the existing classifier logic with your own.

    **Current Implementation (using `timm`):**

    ```python
    # --- In generate_counterfactual.py ---

    classifier = timm.create_model(
        CLASSIFIER_CONFIG["MODEL_NAME"], 
        pretrained=False, 
        num_classes=CLASSIFIER_CONFIG["NUM_CLASSES"]
    ).to(DEVICE)

    # Load your pre-trained weights
    classifier.load_state_dict(
        torch.load(CLASSIFIER_CONFIG["WEIGHTS_PATH"], map_location=DEVICE)
    )
    ```

3.  **Example with a Custom ResNet-50**:

    ```python
    import torchvision.models as models
    import torch.nn as nn

    # --- Replace the above block with this for a custom ResNet-50 ---

    # 1. Instantiate your model
    my_classifier = models.resnet50(weights=None) # Or use pretrained weights

    # 2. Modify the final layer for your number of classes
    num_ftrs = my_classifier.fc.in_features
    my_classifier.fc = nn.Linear(num_ftrs, YOUR_NUM_CLASSES)
    my_classifier = my_classifier.to(DEVICE)

    # 3. Load your trained weights
    my_classifier.load_state_dict(torch.load("path/to/your/resnet50_weights.pth", map_location=DEVICE))
    my_classifier.eval()

    # The DCA framework will now use `my_classifier` for guidance.
    ```

-----

### 📜 Citation

If you find this work useful in your research, please cite our paper:

```bibtex
@article{wang2025diffusion,
  title={Diffusion-based Counterfactual Augmentation: Towards Robust and Interpretable Knee Osteoarthritis Grading},
  author={Wang, Zhe and Ru, Yuhua and Chetouani, Aladine and Shiang, Tina and Chen, Fang and Bauer, Fabian and Zhang, Liping and Hans, Didier and Jennane, Rachid and Palmer, William Ewing and others},
  journal={arXiv preprint arXiv:2506.15748},
  year={2025}
}
```

### 🙏 Acknowledgements

The authors gratefully acknowledge the funding from the Ralph Schlaeger Research Fellowship at Massachusetts General Hospital (MGH), Harvard Medical School (HMS). Data used in this research was obtained from the Osteoarthritis Initiative (OAI) and the Multicenter Osteoarthritis Study (MOST) public-use datasets.
