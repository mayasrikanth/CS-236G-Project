import argparse
from glob import glob
import os
import csv

import numpy as np
import torch as th
import torchvision.transforms as T
from tqdm import trange
import PIL
import wandb

from glide_finetune.glide_finetune import run_glide_finetune_epoch
from glide_finetune.glide_util import load_model, sample
from glide_finetune.loader import TextImageDataset
from glide_finetune import train_util
from glide_finetune.wds_loader_eval import glide_wds_loader


def generate_fakes_nonadv():
    print('GENERATING FAKES WITH NON-ADVERSARIAL GLIDE! \n\n')
    captions_path = './glide_eval_captions.csv'
    freeze_transformer = True
    freeze_diffusion = True
    use_fp16 = False
    activation_checkpointing = False
    enable_upsample = False
    side_x, side_y = 64, 64
    sample_bs = 1  # batch size for inference
    sample_gs = 4.0  # guidance scale for inference
    sample_respacing = '100' # respacing for inference
    image_to_upsample = 'low_res_face.png'
    device = th.device("cpu") if not th.cuda.is_available() else th.device("cuda")
    print("USING DEVICE: ", device)
    # Load model checkpoint
    resume_ckpt = "/home/mayasrikanth/non_adv_glide/glide-finetune/glide_checkpoints/glide-ft-1x2192.pt"
    glide_model, glide_diffusion, glide_options = load_model(
        glide_path=resume_ckpt,
        use_fp16=use_fp16,
        freeze_transformer=freeze_transformer,
        freeze_diffusion=freeze_diffusion,
        activation_checkpointing=activation_checkpointing,
        model_type="base" if not enable_upsample else "upsample",
    )
    glide_model.to(device) # moving to gpu
    glide_model.eval()
    # This should be 0
    number_of_params = sum(x.numel() for x in glide_model.parameters())
    print(f"Number of parameters: {number_of_params}")
    number_of_trainable_params = sum(
        x.numel() for x in glide_model.parameters() if x.requires_grad
    )
    print(f"Trainable parameters: {number_of_trainable_params}")

    # Load captions
    captions = None
    with open(captions_path, newline='') as f:
        reader = csv.reader(f)
        captions = list(reader)

    output_dir_fakes = './nonadv_fake_images'
    os.makedirs(output_dir_fakes, exist_ok=True)

    log = {}
    project_name = 'FID_eval'
    wandb.init(project_name)
    print("Wandb setup.")
    # grab the first 100 captions
    with th.no_grad():
        # Iterate through captions, sampling from glide model, saving outpout.
        for idx, prompt in enumerate(captions):
            #print("PROMPT: ", prompt)
            samples = sample(
                glide_model=glide_model,
                glide_options=glide_options,
                side_x=side_x,
                side_y=side_y,
                prompt=prompt[0],
                batch_size=sample_bs,
                guidance_scale=sample_gs,
                device=device,
                prediction_respacing=sample_respacing,
                #upsample_factor=upsample_factor,
                image_to_upsample=image_to_upsample,
            )
            sample_save_path = os.path.join(output_dir_fakes, f"im_fake_{idx}.png")
            train_util.pred_to_pil(samples).save(sample_save_path)
            wandb.log(
                {
                    **log,
                    "Caption":prompt,
                    "Image Number":idx ,
                    "Sampled Image": wandb.Image(sample_save_path, caption=prompt),
                }
            )



def generate_fakes_adv():
    captions_path = './glide_eval_captions.csv'
    freeze_transformer = True
    freeze_diffusion = True
    use_fp16 = False
    activation_checkpointing = False
    enable_upsample = False
    side_x, side_y = 64, 64
    sample_bs = 1  # batch size for inference
    sample_gs = 4.0  # guidance scale for inference
    sample_respacing = '100' # respacing for inference
    image_to_upsample = 'low_res_face.png'
    device = th.device("cpu") if not th.cuda.is_available() else th.device("cuda")
    print("USING DEVICE: ", device)
    # Load model checkpoint
    resume_ckpt = "/home/mayasrikanth/adversarial_glide/glide-finetune/glide_checkpoints/glide-ft-1x2192.pt"
    glide_model, glide_diffusion, glide_options = load_model(
        glide_path=resume_ckpt,
        use_fp16=use_fp16,
        freeze_transformer=freeze_transformer,
        freeze_diffusion=freeze_diffusion,
        activation_checkpointing=activation_checkpointing,
        model_type="base" if not enable_upsample else "upsample",
    )
    glide_model.to(device) # moving to gpu
    glide_model.eval()
    # This should be 0
    number_of_params = sum(x.numel() for x in glide_model.parameters())
    print(f"Number of parameters: {number_of_params}")
    number_of_trainable_params = sum(
        x.numel() for x in glide_model.parameters() if x.requires_grad
    )
    print(f"Trainable parameters: {number_of_trainable_params}")

    # Load captions
    captions = None
    with open(captions_path, newline='') as f:
        reader = csv.reader(f)
        captions = list(reader)

    output_dir_fakes = './adv_fake_images'
    os.makedirs(output_dir_fakes, exist_ok=True)

    log = {}
    project_name = 'FID_eval'
    wandb.init(project_name)
    print("Wandb setup.")

    with th.no_grad():
        # Iterate through captions, sampling from glide model, saving outpout.
        for idx, prompt in enumerate(captions):
            #print("PROMPT: ", prompt)
            samples = sample(
                glide_model=glide_model,
                glide_options=glide_options,
                side_x=side_x,
                side_y=side_y,
                prompt=prompt[0],
                batch_size=sample_bs,
                guidance_scale=sample_gs,
                device=device,
                prediction_respacing=sample_respacing,
                #upsample_factor=upsample_factor,
                image_to_upsample=image_to_upsample,
            )
            sample_save_path = os.path.join(output_dir_fakes, f"im_fake_{idx}.png")
            train_util.pred_to_pil(samples).save(sample_save_path)
            wandb.log(
                {
                    **log,
                    "Caption":prompt,
                    "Image Number":idx ,
                    "Sampled Image": wandb.Image(sample_save_path, caption=prompt),
                }
            )



# See for FID implementation: https://github.com/GaParmar/clean-fid
def extract_real_data(
    data_dir="./data",
    batch_size=4,#1,
    learning_rate=1e-5,
    adam_weight_decay=0.0,
    side_x=64,
    side_y=64,
    resize_ratio=1.0,
    uncond_p=0.0,
    resume_ckpt="",
    checkpoints_dir="./finetune_checkpoints",
    device='cpu',
    project_name="FID_eval",
    use_captions=True,
    use_webdataset=False,
    image_key=None,
    caption_key=None,
    enable_upsample=False,
    upsample_factor=4

):
    print("extracting real images!")
    if "~" in data_dir: # grab eval data directory
        data_dir = os.path.expanduser(data_dir)
    # if "~" in checkpoints_dir:
    #     checkpoints_dir = os.path.expanduser(checkpoints_dir)

    # Create the real image output directories
    output_dir_reals = './real_eval_images_mini'
    os.makedirs(output_dir_reals, exist_ok=True)

    project_name = 'FID_eval'
    wandb.init(project_name)
    print("Wandb setup.")

    # Data setup
    print("Loading data...")
    if use_webdataset:
        dataset = glide_wds_loader(
            urls=data_dir,
            caption_key=caption_key,
            image_key=image_key,
            enable_image=True,
            enable_text=use_captions,
            enable_upsample=enable_upsample,
            # tokenizer=glide_model.tokenizer,
            ar_lower=0.5,
            ar_upper=2.0,
            min_original_height=side_x * upsample_factor,
            min_original_width=side_y * upsample_factor,
            upscale_factor=upsample_factor,
            nsfw_filter=True,
            similarity_threshold_upper=0.0,
            similarity_threshold_lower=0.5,
            words_to_skip=[],
            dataset_name="laion",  # can be laion, alamy.
        )
    else:
        dataset = TextImageDataset(
            folder=data_dir,
            side_x=side_x,
            side_y=side_y,
            resize_ratio=resize_ratio,
            uncond_p=uncond_p,
            shuffle=True,
            tokenizer=glide_model.tokenizer,
            text_ctx_len=glide_options["text_ctx"],
            use_captions=use_captions,
            enable_glide_upsample=enable_upsample,
            upscale_factor=upsample_factor,  # TODO: make this a parameter
        )

    captions = []
    log = {}
    for idx, tup in enumerate(dataset): # Assuming webdataset
        if idx > 0 and idx % 273 == 0:
            break
        caption, pil_im = tup
        fname = 'im_real_{}.png'.format(idx)
        fpath = os.path.join(output_dir_reals, fname)
        pil_im.save(fpath)
        captions.append(caption)

        wandb.log(
            {
                **log,
                "image number": idx,
                "image": wandb.Image(fpath, caption=caption),
            }
        )
    # Save captions
    print("CAPTIONS \n\n")
    # print(captions[:50])
    fname = './glide_eval_captions_mini.csv'
    with open(fname, "w", newline="") as f:
        writer = csv.writer(f)
        for caption in captions:
            writer.writerow([caption])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-data", type=str, default="./data")
    parser.add_argument("--batch_size", "-bs", type=int, default=4)
    parser.add_argument("--learning_rate", "-lr", type=float, default=2e-5)
    parser.add_argument("--adam_weight_decay", "-adam_wd", type=float, default=0.0)
    parser.add_argument("--side_x", "-x", type=int, default=64)
    parser.add_argument("--side_y", "-y", type=int, default=64)
    parser.add_argument(
        "--resize_ratio", "-crop", type=float, default=0.8, help="Crop ratio"
    )
    parser.add_argument(
        "--uncond_p",
        "-p",
        type=float,
        default=0.2,
        help="Probability of using the empty/unconditional token instead of a caption. OpenAI used 0.2 for their finetune.",
    )
    parser.add_argument(
        "--train_upsample",
        "-upsample",
        action="store_true",
        help="Train the upsampling type of the model instead of the base model.",
    )
    parser.add_argument(
        "--resume_ckpt",
        "-resume",
        type=str,
        default="",
        help="Checkpoint to resume from",
    )
    parser.add_argument(
        "--checkpoints_dir", "-ckpt", type=str, default="./glide_checkpoints/"
    )
    parser.add_argument("--use_fp16", "-fp16", action="store_true")
    parser.add_argument("--device", "-dev", type=str, default="")
    parser.add_argument("--log_frequency", "-freq", type=int, default=100)
    parser.add_argument("--freeze_transformer", "-fz_xt", action="store_true")
    parser.add_argument("--freeze_diffusion", "-fz_unet", action="store_true")
    parser.add_argument("--project_name", "-name", type=str, default="glide-finetune")
    parser.add_argument("--activation_checkpointing", "-grad_ckpt", action="store_true")
    parser.add_argument("--use_captions", "-txt", action="store_true")
    parser.add_argument("--epochs", "-epochs", type=int, default=20)
    parser.add_argument(
        "--test_prompt",
        "-prompt",
        type=str,
        default="a group of skiers are preparing to ski down a mountain.",
    )
    parser.add_argument(
        "--test_batch_size",
        "-tbs",
        type=int,
        default=1,
        help="Batch size used for model eval, not training.",
    )
    parser.add_argument(
        "--test_guidance_scale",
        "-tgs",
        type=float,
        default=1.0,
        help="Guidance scale used during model eval, not training.",
    )
    parser.add_argument(
        "--use_webdataset",
        "-wds",
        action="store_true",
        help="Enables webdataset (tar) loading",
    )
    parser.add_argument(
        "--wds_image_key",
        "-wds_img",
        type=str,
        default="jpg",
        help="A 'key' e.g. 'jpg' used to access the image in the webdataset",
    )
    parser.add_argument(
        "--wds_caption_key",
        "-wds_cap",
        type=str,
        default="txt",
        help="A 'key' e.g. 'txt' used to access the caption in the webdataset",
    )
    parser.add_argument(
        "--wds_dataset_name",
        "-wds_name",
        type=str,
        default="laion",
        help="Name of the webdataset to use (laion or alamy)",
    )
    parser.add_argument("--seed", "-seed", type=int, default=0)
    parser.add_argument(
        "--cudnn_benchmark",
        "-cudnn",
        action="store_true",
        help="Enable cudnn benchmarking. May improve performance. (may not)",
    )
    parser.add_argument(
        "--upscale_factor", "-upscale", type=int, default=4, help="Upscale factor for training the upsampling model only"
    )
    parser.add_argument("--image_to_upsample", "-lowres", type=str, default="low_res_face.png")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # CUDA/CPU setup
    args = parse_args()
    if len(args.device) > 0:
        device = th.device(args.device)
    else:
        device = th.device("cpu") if not th.cuda.is_available() else th.device("cuda")

    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    th.backends.cudnn.benchmark = args.cudnn_benchmark

    for arg in vars(args):
        print(f"--{arg} {getattr(args, arg)}")

    if args.use_webdataset:
        # webdataset uses tars
        data_dir = glob(os.path.join(args.data_dir, "*.tar"))

    generate_adv_fakes = False # for getting adversarial fakes.
    if generate_adv_fakes:
        generate_fakes_adv()
    else:
        generate_fakes_nonadv()
        # extract_real_data(
        #     data_dir=data_dir,
        #     batch_size=4, #args.batch_size,
        #     learning_rate=args.learning_rate,
        #     adam_weight_decay=args.adam_weight_decay,
        #     side_x=args.side_x,
        #     side_y=args.side_y,
        #     resize_ratio=args.resize_ratio,
        #     uncond_p=args.uncond_p,
        #     resume_ckpt=args.resume_ckpt,
        #     checkpoints_dir=args.checkpoints_dir,
        #     device=device,
        #     # log_frequency=args.log_frequency,
        #     # freeze_transformer=args.freeze_transformer,
        #     # freeze_diffusion=args.freeze_diffusion,
        #     project_name=args.project_name,
        #     # activation_checkpointing=args.activation_checkpointing,
        #     use_captions=args.use_captions,
        #     # num_epochs=args.epochs,
        #     # test_prompt=args.test_prompt,
        #     # sample_bs=args.test_batch_size,
        #     # sample_gs=args.test_guidance_scale,
        #     use_webdataset=args.use_webdataset,
        #     image_key=args.wds_image_key,
        #     caption_key=args.wds_caption_key,
        #     enable_upsample=args.train_upsample,
        #     upsample_factor=args.upscale_factor
        #     # image_to_upsample=args.image_to_upsample,
        # )
