# MedSAM++: Automated Multi-Organ Segmentation with Atlas-Guided Prompting, 2.5D Context, and Volumetric Refinement

## Project Metadata
### Authors
- **Team:** Nuren Nafisa(g202427580)
- **Supervisor Name:** Dr. Muzammil Behzad
- **Affiliations:** King Fahd University of Petroleum and Minerals(KFUPM) 
<div align="justify">

## Introduction
<strong>Foundation models</strong> like the Segment Anything Model (SAM) have revolutionized computer vision through <strong>promptable, class-agnostic</strong> segmentation. However, their direct application to medical imaging is hindered by a domain gap: standard SAM lacks <strong>3D spatial awareness</strong>, relies on manual prompts which are unscalable for <strong>volumetric data</strong>, and often struggles with <strong>low-contrast boundaries</strong> typical of soft tissue organs. While MedSAM attempted to bridge this gap, it remains a <strong>2D-slice-based model</strong> that requires significant <strong>computational resources</strong> to fine-tune.
However, directly applying these models to medical imaging presents significant challenges. Medical images, such as abdominal CT and MRI scans, are characterized by several factors that complicate segmentation:
<ul>
<li><u><strong>Complex Data Properties:</strong></u> The images are typically grayscale, volumetric (3D), and exhibit high heterogeneity across different scanners and protocols.</li>

<li><u><strong>Complex Anatomical Boundaries:</strong></u> Multi-organ segmentation requires precise delineation of subtle and complex interfaces (e.g., the pancreas-duodenum boundary).</li>

<li><u><strong>Topological Constraints:</strong></u> Preserving the correct anatomical topology—preventing holes or disconnections in structures—is crucial for clinical validity.</li>
</ul>
We propose a fully  <strong>automated, resource-efficient </strong> pipeline that adapts MedSAM for 3D abdominal CT segmentation. Our approach introduces  <strong>Parameter-Efficient Fine-Tuning (PEFT) </strong> via  <strong>Low-Rank Adaptation (LoRA) </strong>, enabling high-performance adaptation on consumer hardware. We further enhance the model with  <strong>2.5D context </strong>integration and a <strong>novel Boundary-Aware Combo Loss</strong> to improve segmentation precision. Finally, we replace manual interaction with an <strong>Automatic Prompt Generator</strong>, creating a true <strong>"click-free" segmentation tool.</strong>

<div align="justify">  </div>
<img width="2000" height="1458" alt="image" src="https://github.com/user-attachments/assets/a6ad4f82-3bf8-4303-adf0-e4ad5677cfbf" />

<div align="justify">

## Problem Statement
The goal of this project is to achieve accurate multi-organ segmentation in abdominal CT scans while overcoming four critical limitations of current foundation models  <strong>MedSAM </strong>:
<ul>
<li><u><strong>Problem 1:</u> The Fine-Tuning Bottleneck.</strong> Retraining massive Vision Transformers (ViT-B has 90M+ parameters) requires A100 clusters. Freezing the model limits learning, while full fine-tuning leads to overfitting on small medical datasets.
</li>

<li><u><strong>Problem 2:</u>Boundary Ambiguity.</strong> Standard losses (like Dice) focus on global volume overlap but often fail to capture sharp, irregular boundaries for small organs (e.g., pancreas), leading to over-smoothed or "blobby" predictions.</li>

<li><u><strong>Problem 3:</u> The "Human-in-the-Loop" Requirement.</strong> Standard MedSAM is interactive, requiring a human to draw a box for every single slice. This is impractical for clinical workflows involving 3D volumes with hundreds of slices.
</li>

<li><u><strong>Problem 4:</u> Lack of 3D Context.</strong> Standard MedSAM processes images slice-by-slice, ignoring the volumetric relationship between adjacent slices, leading to "flickering" and inconsistent 3D shapes.
</li>
</ul>
<div align="justify">  </div>
<div align="justify">

## Application Area and Project Domain
This work is situated in the domain of medical image analysis, specifically targeting :<strong>automated multi-organ segmentation:</strong> in abdominal computed tomography (CT) scans. The primary clinical and research applications include:
<ul>
<li><u><strong>Automated Volumetric Screening:</strong></u> The pipeline enables rapid, hands-free quantification of organ volumes (liver, kidneys, spleen) from CT scans. This allows for scalable population-level screening and monitoring of organ health without requiring time-consuming manual intervention by radiologists.</li>

<li><u><strong>Surgical Planning and Navigation:</strong></u> By enforcing boundary precision through advanced loss functions, our model generates accurate 3D anatomical maps. These are critical for pre-operative planning, helping surgeons define resection margins and visualize spatial relationships between organs.
</li>

<li><u><strong>Efficient AI Development (Democratization):</strong></u> A key contribution of this project is enabling high-performance medical segmentation on consumer-grade hardware. By employing Parameter-Efficient Fine-Tuning (PEFT) techniques, we demonstrate that powerful foundation models can be adapted for clinical tasks using a single GPU, removing the barrier of requiring massive computing clusters.
</li>
</ul>
A robust, automated method offers a key advantage: generalization across diverse clinical sites with varying imaging protocols with less computation. Furthermore, by incorporating calibration and uncertainty estimation, the pipeline supports a human-in-the-loop workflow by automatically flagging low-confidence slices for expert review, ensuring reliability in critical clinical decision-making.
<div align="justify">  </div>
   
<img width="2037" height="1254" alt="image" src="https://github.com/user-attachments/assets/588388b0-a501-49eb-9cdc-f00bd7ee2bb5" />

<div align="justify">

## What is the paper trying to do, and what are you planning to do?
<strong><ins>What the paper does:</ins>(Paper Link-https://www.nature.com/articles/s41467-024-44824-z#Sec19)</strong>

<strong>Summary of the Baseline Paper:</strong>

<strong>Core Function:</strong> MedSAM  adapts a promptable Vision Transformer (ViT) for medical image segmentation. It adapts the Segment Anything Model (SAM) for medical images by fine-tuning the mask decoder while keeping the massive image encoder frozen.


<strong>Key Strength:</strong>Produces It generalizes well to unseen medical tasks when provided with a valid prompt (bounding box), achieving a "one-model-fits-all" capability.

<strong>Primary Limitations:</strong>
<ul>
<li><u><strong>Relies on manual or simulated prompts, which is not scalable for full-volume segmentation.</strong></u></li>

<li><u><strong>2D Slice Independence, as processes every slice in isolation, ignoring the 3D volumetric context, which leads to "flickering" predictions and inconsistent organ shapes.</strong></u></li>

<li><u><strong>Boundary Ambiguity, as it struggles with low-contrast boundaries (like the pancreas) because standard losses prioritize volume overlap over edge alignment.</strong></u></li>
</ul>
<strong\>What we will do:</strong>
We will enhance the baseline MedSAM into a fully automated, high-precision pipeline through five key technical contributions, re-ordered to follow the data flow pipeline:

<strong>Enhancement 1:</strong> 2.5D Context Integration (Input Preprocessing)

<ul>
<li>\<u><strong>Modify the input pipeline to construct a 2.5D stack of k neighboring slices (e.g., previous, current, and next).</strong></u></li>

<li><u><strong>Feed this multi-slice stack as input to the model's encoder.</strong></u></li>

<li><u><strong>Provide the model with local volumetric cues and depth information to improve organ distinction and reduce inter-slice inconsistencies.</strong></u></li>
</ul>
<strong>Enhancement 2:</strong> Automated Prompt Generator (Automation Before Encoding)
<ul>
<<li><u><strong>Develop a heuristic-based algorithm to automatically detect target organs within the CT slices.</strong></u></li>

<li><u><strong>Apply anatomical priors such as Hounsfield Unit (HU) thresholds, organ size constraints, and aspect ratios to identify candidate regions.</strong></u></li>

<li><u><strong>Generate accurate bounding box prompts from these regions, eliminating the need for manual human intervention.</strong></u></li>
</ul>
<strong>Enhancement 3:</strong> Parameter-Efficient Architecture (LoRA) (Architecture Modification)
<ul>
<li><u><strong>Inject trainable Low-Rank Adaptation (LoRA) layers into the Multi-Head Attention blocks of the Vision Transformer.</strong></u></li>

<li><u><strong>Freeze the original massive encoder parameters to minimize computational requirements.</strong></u></li>

<li><u><strong>Train only the small LoRA layers (updating less than 1% of total parameters) to adapt the model to medical-specific features on standard GPUs.</strong></u></li>
</ul>
<strong\>Enhancement 4:</strong> Boundary-Aware Combo Loss (Training Objective)
<ul>
<li><u><strong>Replace standard loss functions with a specialized ComboLoss: $0.4 \times \text{Dice} + 0.4 \times \text{Focal} + 0.2 \times \text{Boundary}$.</strong></u></li>
<li><u><strong>Incorporate Focal Loss to address class imbalance for smaller or harder-to-segment organs.</strong></u></li>

<li><u><strong>Integrate a Laplacian Boundary Loss to explicitly penalize edge errors, forcing the model to generate sharper and more anatomically accurate contours.</strong></u></li>
</ul>
<strong>Enhancement 5:</strong> 3D Assembly & Refinement (Post-Processing)
<ul>
<li><u><strong>Assemble the slice-by-slice 2D predictions into a coherent 3D volume.</strong></u></li>

<li><u><strong>Apply 3D morphological smoothing and connected components analysis.</strong></u></li>

<li><u><strong>Refine the final 3D output to remove noise, fill gaps, and ensure topological consistency across the organ volume.</strong></u></li>
</ul>
<strong>Evaluation Plan:</strong>

<strong>Quantitative Metrics:</strong> Dice Similarity Coefficient (Dice), Normalized Surface Dice (NSD), Average Surface Distance (ASD), 95th percentile Hausdorff Distance (95HD).

<strong>Qualitative Analysis:</strong> Mask overlays, 3D surface renderings, zoom-ins on complex boundaries, and visualizations of failure cases.

<strong>Zero-Shot Protocol:</strong> A strict separation where no organ labels are used for training or prompt generation; labels are used exclusively for evaluation.

<div align="justify">  </div>
# THE FOLLOWING IS SUPPOSED TO BE DONE LATER

### Project 
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
