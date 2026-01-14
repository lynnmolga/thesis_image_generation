## Code Overview

This repository contains several independent image-generation pipelines developed as part of my M.Sc. thesis.  
Each script represents a different approach to generating visual stimuli.

---

### `stable_diffusion_with_pretrained.py`

Image generation using pretrained Stable Diffusion models (via the Hugging Face / diffusers ecosystem).

This pipeline was used to generate the **final stimulus sets** used in the experiments.  
It supports controlled generation, reproducibility via fixed seeds, and batch image creation.

**Example outputs:**

<img width="128" height="128" alt="3_p5_cs1 7_n2_20250331_185257" src="https://github.com/user-attachments/assets/4b5c247d-eccf-44ff-9b4f-a0354fdd0995" />
<img width="128" height="128" alt="5_outline2_p2_cs1 2_n1_20250512_161638" src="https://github.com/user-attachments/assets/b1190a3c-1f4b-43cc-98ac-043fe062626c" />
<img width="128" height="128" alt="7_outline_p4_cs1 2_n2_20250513_154932" src="https://github.com/user-attachments/assets/7c0383e6-f528-4ea8-8e72-8a69ce51f9fa" />


---

### `gan_landscapes.py`

A custom GAN implementation trained to generate landscape-like images.

This script includes:
- model definition
- training loop
- image sampling

It was used to explore classic generative modeling approaches and to understand the strengths and limitations of GAN-based generation for this task.

**Example outputs:**

<img width="128" height="128" alt="landscape_digit1_variant1_00000814_(3)" src="https://github.com/user-attachments/assets/7e51cd95-2f5d-497e-8a89-1c6b6202a5f3" />
<img width="128" height="128" alt="sample_epoch_600_img_2" src="https://github.com/user-attachments/assets/9fa2bba9-2fe0-4f8a-936a-ca248ba9e39f" />
<img width="128" height="128" alt="sample_epoch_49000_img_0" src="https://github.com/user-attachments/assets/df645ba8-b098-4564-b3da-be57c659efab" /> 


---

### `stable_diffusion_from_scratch.py`

A diffusion-based image generation model implemented from scratch.

The model produces meaningful image samples and was used to explore diffusion processes and training dynamics. Pretrained Stable Diffusion models were used for large-scale generation in the final experiments due to computational resource limitations.

**Example outputs:**

<img width="128" height="128" alt="epoch_055digit6" src="https://github.com/user-attachments/assets/dcf2be93-893a-417d-8a64-429158666fdd" />
<img width="128" height="128" alt="epoch_055digit8" src="https://github.com/user-attachments/assets/55481344-5df1-4c8c-bd68-4c3da0d3ee87" />
<img width="128" height="128" alt="epoch_044digit6" src="https://github.com/user-attachments/assets/84647c9f-3a9f-4319-8f47-cda60c4735f5" />

