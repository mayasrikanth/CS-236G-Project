# CS-236G-Project: Illustrating Stories with Deep Generative Models


For the initial milestone, I downloaded off-the-shelf models (GPT-J, GLIDE) on a GCP instance and issued various prompts to them, so there are no specific scripts in this repository that are required to replicate the Milestone 1 results. The instructions below describe how I generated results in the writeup:


**Instructions for Replicating Milestone 1 Zero-shot Baseline Results:**

(1) **Generating prompts:** to generate the prompts, I used EleutherAI's 6-billion parameter GPT-J model, trained on a cross entropy loss objective to maximize the likelihood of predicting the correct next token. I used their [playground site](https://6b.eleuther.ai/) to generate the prompts discussed in my writeup. I also supplied my own prompts to inspire creative outputs from GPT-J. All prompts used for the baseline analysis are included in this repo in the file prompts-initial-dataset.txt. 

(2) **Generating images:** I cloned openAI's [glide-text2im](https://github.com/openai/glide-text2im) repository and utilized their small, filtered GLIDE model (a CLIP-guided Gaussian diffusion model) to produce images given various prompts. Specifically, I used their colab demo [glide-text2im](https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb). 




For future deadlines, I plan to include well-documented/user-friendly colab notebook for generating novel story prompts and creating illustrations based on these prompts!
