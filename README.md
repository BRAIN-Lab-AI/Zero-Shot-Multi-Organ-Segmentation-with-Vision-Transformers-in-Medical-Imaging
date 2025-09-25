# Zero-Shot-Multi-Organ-Segmentation-with-Vision-Transformers-in-Medical-Imaging

## Project Metadata
### Authors
- **Team:** Nuren Nafisa(g202427580)
- **Supervisor Name:** Dr. Muzammil Behzad
- **Affiliations:** SABIC, ARAMCO and KFUPM (write your institution name, and/or KFUPM)
<div align="justify">

## Introduction
<strong>Foundation models</strong> like the Segment Anything Model (SAM) have introduced powerful, <strong>class-agnostic</strong> segmentation using <strong>Vision Transformers (ViTs)</strong>. These models can generate high-quality masks from simple prompts (e.g., a point or bounding box) without any task-specific training. However, directly applying these models to medical imaging presents significant challenges. Medical images, such as abdominal CT and MRI scans, are characterized by several factors that complicate segmentation:

<ul>
<li><u><strong>Complex Data Properties:</strong></u> The images are typically grayscale, volumetric (3D), and exhibit high heterogeneity across different scanners and protocols.</li>

<li><u><strong>Complex Anatomical Boundaries:</strong></u> Multi-organ segmentation requires precise delineation of subtle and complex interfaces (e.g., the pancreas-duodenum boundary).</li>

<li><u><strong>Topological Constraints:</strong>strong></u> Preserving the correct anatomical topology—preventing holes or disconnections in structures—is crucial for clinical validity.</li>
</ul>
We propose a <strong>zero-shot pipeline</strong> that adapts promptable ViTs for multi-organ segmentation without any organ-specific training. Our method converts weak anatomical priors—derived from atlas registration and simple image heuristics—into <strong>automatic prompts</strong>. To address the lack of 3D context in standard 2D ViTs, we introduce a <strong>2.5D input</strong> by stacking adjacent slices, providing the model with local volumetric cues. Finally, we assemble the 2D segmentations into a 3D volume and apply <strong>topology-aware and boundary-aware refinement</strong> to ensure anatomical plausibility and consistency. This training-free approach aims to reduce the reliance on large, annotated datasets while maintaining robust performance.
<div align="justify">  </div>
<img width="2000" height="1458" alt="image" src="https://github.com/user-attachments/assets/a6ad4f82-3bf8-4303-adf0-e4ad5677cfbf" />

<div align="justify">

## Problem Statement
The goal of this project is to achieve accurate multi-organ segmentation in abdominal CT/MR scans without using any organ-specific labels for training. This zero-shot objective is hindered by three primary problems:
<ul>
<li><u><strong>Problem 1:</u> Generating high-quality prompts automatically is difficult.</strong> Promptable models require accurate initial cues. In a zero-shot setting, without labeled data to learn from, we must rely on weak, unsupervised priors like atlas alignment and intensity heuristics, which can be noisy and imprecise.</li>

<li><u><strong>Problem 2:</u> Slice-wise 2D decoding lacks 3D consistency.</strong> Applying a ViT independently on each slice ignores the volumetric nature of the data. This leads to slice-to-slice flickering, incoherent 3D shapes, and failures at ambiguous boundaries where adjacent slice information is critical.</li>

<li><u><strong>Problem 3:</u> Thin structures and small organs suffer from topological errors.</strong> Standard segmentation losses do not explicitly preserve connectivity. Consequently, thin, branching structures like blood vessels or small organs like the pancreas are often fragmented or merged with adjacent tissues, violating anatomical plausibility.</li>
</ul>
<div align="justify">  </div>
<div align="justify">

## Application Area and Project Domain
This work is situated in the domain of medical image analysis, specifically targeting multi-organ segmentation in abdominal computed tomography (CT) and magnetic resonance imaging (MRI) scans. The primary clinical and research applications include:

Surgical Planning and Navigation: Precise 3D models of organs like the liver and kidneys are critical for pre-operative planning and intra-operative guidance, helping to define resection margins and avoid critical structures.

Disease Quantification and Follow-up: Accurate segmentation enables the volumetric measurement of organs for tracking tumor growth, assessing treatment response, and monitoring chronic conditions over time.

Automated Dataset Curation: The pipeline can rapidly generate preliminary segmentations for new datasets, significantly reducing the manual annotation burden required to train and validate fully supervised models.

A robust, zero-shot method offers a key advantage: generalization across diverse clinical sites with varying imaging protocols without requiring retraining. Furthermore, by incorporating calibration and uncertainty estimation, the pipeline supports a human-in-the-loop workflow by automatically flagging low-confidence slices for expert review, ensuring reliability in critical clinical decision-making.

<img width="2037" height="1254" alt="image" src="https://github.com/user-attachments/assets/588388b0-a501-49eb-9cdc-f00bd7ee2bb5" />
<div align="justify">  </div>
<div align="justify">

## What is the paper trying to do, and what are you planning to do?
What the paper does: 
A. Summary of the Baseline Paper (What it does):

Core Function: Adapts a promptable Vision Transformer (ViT) for medical image segmentation.

Key Strength: Produces high-quality, class-agnostic 2D masks from simple user-provided prompts (points or bounding boxes).

Primary Limitations:

Relies on manual or simulated prompts, which is not scalable for full-volume segmentation.

Processes each slice independently, leading to a lack of 3D consistency and slice-to-slice flickering.

Shows reduced reliability on thin/branching structures and small organs due to the absence of topological constraints.

What we will do:
We will enhance this baseline into a complete zero-shot multi-organ segmentation pipeline through three key, reproducible enhancements:

Enhancement 1: Automatic Prompt Generation

Replace manual prompts with atlas-guided search windows to locate organs.

Apply CT/MR-specific heuristics (e.g., HU thresholding, connected components analysis) to generate candidate bounding boxes or points.

Implement label-free filtering based on size, location, and intensity consistency.

Enhancement 2: Incorporation of 2.5D Context

Stack k neighboring slices as input channels to the ViT encoder.

Provide local volumetric cues to the model without changing the core architecture.

Aim to improve boundary decisions and reduce inter-slice inconsistencies.

Enhancement 3: 3D Assembly and Refinement

Link 2D slice masks into coherent 3D volumes using inter-slice tracking (e.g., IoU-based).

Apply post-processing refinement (e.g., morphological operations, boundary smoothing) to ensure anatomical plausibility and sharp edges.

C. Evaluation Plan:

Quantitative Metrics: Dice Similarity Coefficient (Dice), Normalized Surface Dice (NSD), Average Surface Distance (ASD), 95th percentile Hausdorff Distance (95HD).

Qualitative Analysis: Mask overlays, 3D surface renderings, zoom-ins on complex boundaries, and visualizations of failure cases.

Zero-Shot Protocol: A strict separation where no organ labels are used for training or prompt generation; labels are used exclusively for evaluation.

<div align="justify">  </div>
# THE FOLLOWING IS SUPPOSED TO BE DONE LATER

### Project Documents
- **Presentation:** [Project Presentation](/presentation.pptx)
- **Report:** [Project Report](/report.pdf)

### Reference Paper
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)

### Reference Dataset
- [LAION-5B Dataset](https://laion.ai/blog/laion-5b/)


## Project Technicalities

### Terminologies
- **Diffusion Model:** A generative model that progressively transforms random noise into coherent data.
- **Latent Space:** A compressed, abstract representation of data where complex features are captured.
- **UNet Architecture:** A neural network with an encoder-decoder structure featuring skip connections for better feature preservation.
- **Text Encoder:** A model that converts text into numerical embeddings for downstream tasks.
- **Perceptual Loss:** A loss function that measures high-level differences between images, emphasizing perceptual similarity.
- **Tokenization:** The process of breaking down text into smaller units (tokens) for processing.
- **Noise Vector:** A randomly generated vector used to initialize the diffusion process in generative models.
- **Decoder:** A network component that transforms latent representations back into image space.
- **Iterative Refinement:** The process of gradually improving the quality of generated data through multiple steps.
- **Conditional Generation:** The process where outputs are generated based on auxiliary inputs, such as textual descriptions.

### Problem Statements
- **Problem 1:** Achieving high-resolution and detailed images using conventional diffusion models remains challenging.
- **Problem 2:** Existing models suffer from slow inference times during the image generation process.
- **Problem 3:** There is limited capability in performing style transfer and generating diverse artistic variations.

### Loopholes or Research Areas
- **Evaluation Metrics:** Lack of robust metrics to effectively assess the quality of generated images.
- **Output Consistency:** Inconsistencies in output quality when scaling the model to higher resolutions.
- **Computational Resources:** Training requires significant GPU compute resources, which may not be readily accessible.

### Problem vs. Ideation: Proposed 3 Ideas to Solve the Problems
1. **Optimized Architecture:** Redesign the model architecture to improve efficiency and balance image quality with faster inference.
2. **Advanced Loss Functions:** Integrate novel loss functions (e.g., perceptual loss) to better capture artistic nuances and structural details.
3. **Enhanced Data Augmentation:** Implement sophisticated data augmentation strategies to improve the model’s robustness and reduce overfitting.

### Proposed Solution: Code-Based Implementation
This repository provides an implementation of the enhanced stable diffusion model using PyTorch. The solution includes:

- **Modified UNet Architecture:** Incorporates residual connections and efficient convolutional blocks.
- **Novel Loss Functions:** Combines Mean Squared Error (MSE) with perceptual loss to enhance feature learning.
- **Optimized Training Loop:** Reduces computational overhead while maintaining performance.

### Key Components
- **`model.py`**: Contains the modified UNet architecture and other model components.
- **`train.py`**: Script to handle the training process with configurable parameters.
- **`utils.py`**: Utility functions for data processing, augmentation, and metric evaluations.
- **`inference.py`**: Script for generating images using the trained model.

## Model Workflow
The workflow of the Enhanced Stable Diffusion model is designed to translate textual descriptions into high-quality artistic images through a multi-step diffusion process:

1. **Input:**
   - **Text Prompt:** The model takes a text prompt (e.g., "A surreal landscape with mountains and rivers") as the primary input.
   - **Tokenization:** The text prompt is tokenized and processed through a text encoder (such as a CLIP model) to obtain meaningful embeddings.
   - **Latent Noise:** A random latent noise vector is generated to initialize the diffusion process, which is then conditioned on the text embeddings.

2. **Diffusion Process:**
   - **Iterative Refinement:** The conditioned latent vector is fed into a modified UNet architecture. The model iteratively refines this vector by reversing a diffusion process, gradually reducing noise while preserving the text-conditioned features.
   - **Intermediate States:** At each step, intermediate latent representations are produced that increasingly capture the structure and details dictated by the text prompt.

3. **Output:**
   - **Decoding:** The final refined latent representation is passed through a decoder (often part of a Variational Autoencoder setup) to generate the final image.
   - **Generated Image:** The output is a synthesized image that visually represents the input text prompt, complete with artistic style and detail.

## How to Run the Code

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/enhanced-stable-diffusion.git
    cd enhanced-stable-diffusion
    ```

2. **Set Up the Environment:**
    Create a virtual environment and install the required dependencies.
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3. **Train the Model:**
    Configure the training parameters in the provided configuration file and run:
    ```bash
    python train.py --config configs/train_config.yaml
    ```

4. **Generate Images:**
    Once training is complete, use the inference script to generate images.
    ```bash
    python inference.py --checkpoint path/to/checkpoint.pt --input "A surreal landscape with mountains and rivers"
    ```

## Acknowledgments
- **Open-Source Communities:** Thanks to the contributors of PyTorch, Hugging Face, and other libraries for their amazing work.
- **Individuals:** Special thanks to bla, bla, bla for the amazing team effort, invaluable guidance and support throughout this project.
- **Resource Providers:** Gratitude to ABC-organization for providing the computational resources necessary for this project.
