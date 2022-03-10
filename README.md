# CS-236G-Project: Illustrating Stories with Deep Generative Models


For the initial milestone, I downloaded off-the-shelf models (GPT-J, GLIDE) on a GCP instance and issued various prompts to them, so there are no specific scripts in this repository that are required to replicate the Milestone 1 results. The instructions below describe how I generated results in the writeup:


**Instructions for Replicating Milestone 1 Zero-shot Baseline Results, Milestone 2 Results, Final Results:**

(1) (Milestone 1) **Generating prompts:** to generate the prompts, I used EleutherAI's 6-billion parameter GPT-J model, trained on a cross entropy loss objective to maximize the likelihood of predicting the correct next token. I used their [playground site](https://6b.eleuther.ai/) to generate the prompts discussed in my writeup. I also supplied my own prompts to inspire creative outputs from GPT-J. All prompts used for the baseline analysis are included in this repo in the file prompts-initial-dataset.txt. For next steps, I've started a colab notebook (in progress) for generating story prompts with a GPT model. 

(2) (Milestone 1)**Generating images:** I cloned openAI's [glide-text2im](https://github.com/openai/glide-text2im) repository and utilized their small, filtered GLIDE model (a CLIP-guided Gaussian diffusion model) to produce images given various prompts. Specifically, I used their colab demo [glide-text2im](https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb). For convenience, I've included a notebook in this repo which installs the necessary dependencies to run GLIDE's text2im code in the browser. 

(3) (Milestone 2) **Refining Prompt Tuning Pipeline:** I implemented and tested sequential prompt-tuning pipeline to process input text and
create an image description, as well as generate novel ideas, with rough code included in the "generate story prompts" colab notebook. I relied on a combination of GPT-Neo (~5GB, loaded onto GCP GPU) and GPT-J, which was too large to load ~12GB (I ended up using their playground site due to compute limitations). I test a pipeline of 3 prompt tuning tasks: (1) Paraphrase with entity linking, (2) Add visual desciptors, and (3) Generate creative endings, explained in more detail in the milestone submission. 

(4) (Milestone 2) **Probe artistic capabilities of GLIDE and v-diffusion**: For this step, I cloned openAI's [glide-text2im](https://github.com/openai/glide-text2im) and Katherine Crowson's [v-diffusion-pytorch](https://github.com/crowsonkb/v-diffusion-pytorch), and issued creative prompts to both models. 


(5) (Milestone 2) **In-painting: test GLIDE in-painting for combining the outputs of 2 diffusion models:** Specifically, I used openAI's inpainting Colabs (I modified the input images, prompts, and mask shapes) to enable GLIDE in-painting over scenery produced by v-diffusion. 

(6) **FINAL PROJECT RESULTS CODE**
- I constructed novel story prompts for GPT-3 (gpt3-complete-prompts.txt, gpt3-partial-prompts.txt)
- I downloaded a subset of CC12M in webdataset format using [img2dataset](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/mscoco.md)
- I adapted fine-tuning scripts from [glide-finetune](https://github.com/afiaka87/glide-finetune) and wrote new code for fine-tuning with novel adversarial objective (Discriminator.py, train_glide_adversarial.py, finetune_glide_adversarial.py), as well as code for loading data for FID calculation with [clean-fid](https://github.com/GaParmar/clean-fid) (FID_eval.py, wds_loader_eval.py).



For the final deadline, I plan to include well-documented/user-friendly colab notebook for generating novel story prompts and creating illustrations based on these prompts!
