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
<strong><ins>What the paper does:</ins></strong>

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
<li><u><strong>Modify the input pipeline to construct a 2.5D stack of k neighboring slices (e.g., previous, current, and next).</strong></u></li>

<li><u><strong>Feed this multi-slice stack as input to the model's encoder.</strong></u></li>

<li><u><strong>Provide the model with local volumetric cues and depth information to improve organ distinction and reduce inter-slice inconsistencies.</strong></u></li>
</ul>
<strong>Enhancement 2:</strong> Automated Prompt Generator (Automation Before Encoding)
<ul>
<li><u><strong>Develop a heuristic-based algorithm to automatically detect target organs within the CT slices.</strong></u></li>

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

<strong>Quantitative Metrics:</strong> 
<ul>
<li><u><strong>Dice Similarity Coefficient (DSC):</strong> Measures volume overlap.</u></li>
<li><u><strong>Normalized Surface Dice (NSD):</strong> Measures boundary accuracy (critical for surgery).</u></li>
<li><u><strong>Comparison:</strong> Baseline MedSAM vs. Improved MedSAM vs. Specialist U-Net.</u></li>
</ul>


<strong>Qualitative Analysis:</strong>
<ul>
<li><u><strong>Side-by-side visualization of predictions.</strong></u></li>
<li><u><strong>Error maps (Green/Red/Blue) showing leakage and missed regions.</strong></u></li>
</ul>

<strong>Validation Strategy:</strong>
<ul>
<li><u><strong>Internal:</strong> FLARE22 Test Set (5 held-out scans).</u></li>
<li><u><strong>External:</strong> AMOS Dataset (Zero-shot transfer to new data/scanner).</u></li>


<div align="justify">  </div>

### Project 
- **Presentation Slide:** [Presentation Slide](/report.pdf)
 
- **Report:** [Project Report](/report.pdf)

### Reference Paper
- [Segment anything in medical images](https://www.nature.com/articles/s41467-024-44824-z#Sec19)

### Reference Dataset
- [MICCAI FLARE22 Challenge Dataset (50 Labeled Abdomen CT Scans)](https://zenodo.org/records/7860267)


## Project Technicalities

### Terminologies

- **MedSAM++:** An enhanced medical image segmentation pipeline that builds upon MedSAM by adding full automation, 2.5D context, and boundary-aware learning for abdominal CT scans.
- **2.5D Context:** An input preprocessing technique where each slice is processed alongside its adjacent neighbors (tri-slice stack [I₍ᵢ₋₁₎, Iᵢ, I₍ᵢ₊₁₎]) to provide local depth cues without full 3D modeling.
- **Atlas-Guided Prompt Generator:** A deterministic module that automatically generates bounding box prompts using anatomical priors, HU thresholding, and morphological operations, eliminating manual input.
- **Low-Rank Adaptation (LoRA):** A parameter-efficient fine-tuning method that inserts trainable rank-decomposition matrices into transformer attention blocks, updating <1% of parameters.
- **Parameter-Efficient Fine-Tuning (PEFT):** A strategy that updates only a small subset of model parameters during adaptation to new domains, significantly reducing computational requirements.
- **Boundary-Aware Combo Loss:** A composite loss function combining Dice loss, Focal loss, and a Laplacian boundary term to improve contour accuracy alongside regional overlap.
- **Click-Free Automation:** A fully automated segmentation pipeline that requires no manual prompts or user interaction, enabling scalable 3D volume processing.
- **Volumetric Refinement:** A post-processing step that assembles 2D slice predictions into 3D volumes and applies morphological operations to ensure topological consistency.
- **Tri-Slice Stack:** An input representation consisting of three consecutive axial slices that provides limited 3D context to a 2D segmentation model.
- **Laplacian Boundary Loss:** A contour-aware loss term that uses a Laplacian kernel to penalize discrepancies between predicted and ground truth boundaries.
- **Normalized Surface Dice (NSD):** A boundary-focused evaluation metric that measures the fraction of surface points within a specified tolerance (e.g., 2mm) between prediction and ground truth.
- **Zero-Shot External Validation:** Testing a model on completely unseen datasets from different sources without any fine-tuning, assessing its generalization capability.
- **Morphological Post-Processing:** Volumetric operations like connected-component filtering, hole-filling, and binary closing used to refine 3D segmentation masks.
- **Domain Shift:** The performance degradation that occurs when a model trained on one dataset (e.g., FLARE22) is applied to data from different sources (e.g., AMOS) with varying scanners or protocols.

### Problem Statements
- **Problem 1:** Full fine-tuning of massive models like ViT-B leads to overfitting on small medical datasets, while freezing parameters limits learning.
- **Problem 2:** Standard segmentation losses often fail to capture sharp, irregular boundaries for small organs, resulting in over-smoothed and inaccurate predictions.
- **Problem 3:** The requirement for manual annotation on every slice makes the standard MedSAM model impractical for clinical 3D volumes.
- **Problem 4:**  Processing images slice-by-slice ignores volumetric context, leading to inconsistent 3D shapes and slice-to-slice "flickering.

### Loopholes or Research Areas
- **Heuristic Prompting Limitations:** The atlas-guided prompt generator relies on fixed rules and HU thresholds, which may fail with atypical anatomies, post-surgical changes, or unusual organ morphologies, creating a need for learned prompt proposal networks.
- **Limited 3D Context:** The 2.5D tri-slice input only captures short-range spatial context, lacking long-range volumetric reasoning and potentially struggling with globally inconsistent shapes across a full 3D volume.
- **Boundary Loss Sensitivity:** The Laplacian boundary loss can be sensitive to annotation noise and tolerance parameter choices, potentially amplifying errors from imperfect ground truth labels.
- **Hounsfield Unit (HU) Thresholding:** A preprocessing technique that clips CT intensity values to specific ranges to enhance contrast for abdominal organs.
- **Domain Shift Vulnerability:** Despite improved generalization, performance may still degrade significantly across different scanner manufacturers, imaging protocols, or patient populations not seen during training.
- **Computational and Data Efficiency:** While LoRA reduces parameters, the ViT backbone remains large, and the model's data hunger for small organ segmentation (e.g., pancreas) persists, highlighting needs in few-shot and self-supervised learning.
- **Post-Processing Dependence:** The reliance on morphological post-processing (e.g., connected components) indicates the core model may still produce fragmented outputs, suggesting underlying segmentation coherence issues.
- **Clinical Integration Gap:** The pipeline lacks quality assurance (QA) mechanisms and uncertainty estimation, which are critical for real-world clinical deployment and trust.
- **Limited Pathology Handling:** The model is primarily designed and tested on healthy organ anatomy, leaving its performance on pathological cases (e.g., tumors, lesions) largely unverified.

### Problem vs. Ideation: Proposed Ideas to Solve the Problems

1.  **Parameter-Efficient Fine-Tuning via LoRA:** Address the fine-tuning bottleneck by integrating Low-Rank Adaptation (LoRA) into the ViT encoder, updating 1% of parameters to enable effective adaptation on single GPU systems while preventing overfitting.
2.  **Boundary-Aware Combo Loss:** Solve boundary ambiguity by combining Dice loss, Focal loss, and a novel Laplacian boundary term in a composite loss function that explicitly penalizes contour errors for sharper organ delineation.
3.  **Atlas-Guided Automatic Prompt Generator:** Eliminate human-in-the-loop requirements by developing a rule-based module that generates bounding box prompts automatically using anatomical priors, HU thresholding, and morphological operations.
4.  **2.5D Context Integration:** Overcome the lack of 3D context by processing tri-slice stacks [I₍ᵢ₋₁₎, Iᵢ, I₍ᵢ₊₁₎] as input to provide local volumetric information and improve inter-slice consistency.
5.  **Volumetric Refinement Pipeline:** Ensure 3D topological consistency through post-processing with connected-component filtering, hole-filling, and morphological operations to transform 2D predictions into coherent 3D structures.
6.  **Cross-Dataset Generalization Framework:** Enhance domain robustness through rigorous internal (FLARE22) and external (AMOS) validation protocols, with provisions for future test-time adaptation to address domain shift.

### Proposed Solution: Code-Based Implementation
This repository provides an implementation of the enhanced MedSAM++ pipeline using PyTorch. The solution includes:
- **LoRA-Enhanced ViT Encoder:** Integrates Low-Rank Adaptation modules into the Vision Transformer's attention mechanisms for parameter-efficient fine-tuning, updating <1% of weights while maintaining pre-trained knowledge.
- **2.5D Context Processing:** Implements tri-slice input stacks [I₍ᵢ₋₁₎, Iᵢ, I₍ᵢ₊₁₎] with efficient channel concatenation to provide volumetric context to the 2D segmentation model.
- **Atlas-Guided Prompt Generator:** Develops an automated bounding box proposal system using Hounsfield Unit thresholding, morphological operations, and connected-component analysis to eliminate manual prompting.
- **Boundary-Aware Combo Loss:** Combines Dice loss, Focal loss, and Laplacian boundary regularization in a weighted objective function to enforce precise contour delineation alongside regional overlap.
- **Volumetric Post-Processing:** Implements 3D connected-component filtering, hole-filling, and morphological closing operations to transform 2D slice predictions into topologically consistent 3D organ masks.

### Key Components

<strong>Core Components</strong>
- **pre_CT_MR_split.py**- Data preprocessing and train/val/test split with HU windowing
- **train_unet.py**- Specialist U-Net baseline training with BasicUNet architecture
- **train_one_gpu_frozen.py**- MedSAM baseline training with frozen ViT encoder
- **train_one_gpu_unfrozen.py**- MedSAM baseline training with partially unfrozen encoder
- **unfrozen_Improved-ulta_na.py**- Main MedSAM++ training with 2.5D context and LoRA
- 
<strong>Evaluation & Visualization</strong>
- **2_BaseLine_and_U-net_Evaluation(Quantitative).py**- Baseline model evaluation (Dice/NSD)
- **2_Improved_Evaluation(Quantitative).py**- MedSAM++ comprehensive validation
- **Qualitative_Result.py**- 2D error maps and 3D surface rendering

## Model Workflow
The workflow of the Enhanced Stable Diffusion model is designed to translate textual descriptions into high-quality artistic images through a multi-step diffusion process:

- **Input:** Raw CT Volume → Preprocessing (Windowing/Resizing).
- **Context:** Extract 2.5D slice stacks.
- **Prompt:** Auto-generator scans slice → Bounding Box.
- **Inference:** MedSAM (with LoRA) predicts 2D mask.
- **Assembly:** Stack 2D masks → 3D Volume → Morphological Refinement → Final Output.

## How to Run the Code

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/BRAIN-Lab-AI/Zero-Shot-Multi-Organ-Segmentation-with-Vision-Transformers-in-Medical-Imaging.git
    cd Zero-Shot-Multi-Organ-Segmentation-with-Vision-Transformers-in-Medical-Imaging
    ```

2. **Set Up the Environment:**
    Create a virtual environment and install the required dependencies.
    ```bash
    # Create virtual environment
    python3 -m venv medsam_env
    source medsam_env/bin/activate  # On Windows: medsam_env\Scripts\activate
    pip install -r requirements.txt
    ```
3. **Data Preprocessing**
   ```bash
   python pre_CT_MR_split.py
    ```
      
4. **Train the Model:**
    1. Download Pre-trained Weights
   ```bash
    mkdir -p work_dir/SAM
    ```
   2. Specialist U-Net Baseline
    ```bash
    python train_unet.py \
  -tr_npy_path data/npy/CT_Abd/train \
  -val_npy_path data/npy/CT_Abd/val \
  -num_epochs 60 \
  -batch_size 4
    ```
    3. MedSAM Baselines
   ```bash
  # Frozen encoder
  python train_one_gpu_frozen.py \
  -tr_npy_path data/npy/CT_Abd/train \
  -val_npy_path data/npy/CT_Abd/val \
  -checkpoint work_dir/SAM/sam_vit_b_01ec64.pth
  -num_epochs 60 \
  -batch_size 4
 # Partially unfrozen encoder  
  python train_one_gpu_unfrozen.py \
  -tr_npy_path data/npy/CT_Abd/train \
  -val_npy_path data/npy/CT_Abd/val \
  -checkpoint work_dir/SAM/sam_vit_b_01ec64.pth
  -num_epochs 60 \
  -batch_size 4
```bash

4. MedSAM++ (Enhanced)
    Once training is complete, use the inference script to generate images.
    ```bash
   # Frozen encoder
   python unfrozen_Improved.py \
   -tr_npy_path data/npy/CT_Abd/train \
  -val_npy_path data/npy/CT_Abd/val \
  -num_epochs 60 \
  -batch_size 4
    ```
   ```bash
   # Partially unfrozen encoder  
   python frozen_Improved.py \
  -tr_npy_path data/npy/CT_Abd/train \
  -val_npy_path data/npy/CT_Abd/val \
  -num_epochs 60 \
  -batch_size 4
    ```

## Acknowledgments
- **Open-Source Communities:** Thanks to the contributors of PyTorch, Hugging Face, and other libraries for their amazing work.
- **Individuals:** Special thanks to bla, bla, bla for the amazing team effort, invaluable guidance and support throughout this project.
- **Resource Providers:** Gratitude to ABC-organization for providing the computational resources necessary for this project.
