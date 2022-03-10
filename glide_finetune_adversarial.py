import os
from typing import Tuple

import torch as th
from glide_text2im.respace import SpacedDiffusion
from glide_text2im.text2im_model import Text2ImUNet
from tqdm import tqdm
from wandb import wandb


from glide_finetune import glide_util, train_util, Discriminator

# from ..train_glide_adversarial import Discriminator

'''Attempting adversarial loss term. '''
# # Original OpenAI implementation
# def q_sample(self, x_start, t, noise=None):
#         """
#         Diffuse the data for a given number of diffusion steps.
#         In other words, sample from q(x_t | x_0).
#         :param x_start: the initial data batch.
#         :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
#         :param noise: if specified, the split-out normal noise.
#         :return: A noisy version of x_start.
#         """
#         if noise is None:
#             noise = th.randn_like(x_start)
#         assert noise.shape == x_start.shape
#         return (
#             _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
#             + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
#         )

# Taken from GLIDE repo
def _extract_into_tensor_adv(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + th.zeros(broadcast_shape, device=timesteps.device)

# Function for extracing original image given predicted noise
# (does the opposite of q_sample)
def denoise_pred(glide_diffusion, x_t, t, pred_noise=None):
        """
        De-diffuse the data for a given number of diffusion steps.
        :param x_t: the data batch after noising in train_step.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, noise predicted by diffusion model ntwork.
        :return: Prediction of x_start given predicted noise.
        Used for training discriminator.
        """
        assert pred_noise != None
        assert pred_noise.shape == x_t.shape
        sqrt_alphas_cumprod = glide_diffusion.sqrt_alphas_cumprod
        sqrt_one_minus_alphas_cumprod = glide_diffusion.sqrt_one_minus_alphas_cumprod

        x_start_num = x_t - _extract_into_tensor_adv(sqrt_one_minus_alphas_cumprod, t, x_t.shape)*pred_noise
        x_start_den = _extract_into_tensor_adv(sqrt_alphas_cumprod, t, x_t.shape)
        x_start_pred = th.div(x_start_num, x_start_den) # element-wise division
        # OR (if ^ doesn't work), try x_start_pred = x_start_num * 1 / _extract_into_tensor(sqrt_alphas_cumprod, t, x_t.shape)
        return x_start_pred



def base_train_step(
    glide_model: Text2ImUNet,
    glide_diffusion: SpacedDiffusion,
    batch: Tuple[th.Tensor, th.Tensor, th.Tensor],
    device: str,
):
    """
    Perform a single training step.

        Args:
            glide_model: The model to train.
            glide_diffusion: The diffusion to use.
            batch: A tuple of (tokens, masks, reals) where tokens is a tensor of shape (batch_size, seq_len), masks is a tensor of shape (batch_size, seq_len) and reals is a tensor of shape (batch_size, 3, side_x, side_y) normalized to [-1, 1].
            device: The device to use for getting model outputs and computing loss.
        Returns:
            The loss.
    """
    tokens, masks, reals = [x.to(device) for x in batch]
    timesteps = th.randint(
        0, len(glide_diffusion.betas) - 1, (reals.shape[0],), device=device
    )
    noise = th.randn_like(reals, device=device)
    x_t = glide_diffusion.q_sample(reals, timesteps, noise=noise).to(device)
    _, C = x_t.shape[:2]
    model_output = glide_model(
        x_t.to(device),
        timesteps.to(device),
        tokens=tokens.to(device), # encodes text prompt 
        mask=masks.to(device),
    )
    epsilon, _ = th.split(model_output, C, dim=1) # predicted noise

    # Find "denoised" x_0
    x0_pred = denoise_pred(glide_diffusion, x_t, timesteps, pred_noise=epsilon)

    # Calculating diffusion network loss
    return (th.nn.functional.mse_loss(epsilon, noise.to(device).detach()), x0_pred, reals)

def upsample_train_step(
    glide_model: Text2ImUNet,
    glide_diffusion: SpacedDiffusion,
    batch: Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor],
    device: str,
):
    """
    Perform a single training step.

        Args:
            glide_model: The model to train.
            glide_diffusion: The diffusion to use.
            batch: A tuple of (tokens, masks, low_res, high_res) where
                - tokens is a tensor of shape (batch_size, seq_len),
                - masks is a tensor of shape (batch_size, seq_len) with dtype torch.bool
                - low_res is a tensor of shape (batch_size, 3, base_x, base_y), normalized to [-1, 1]
                - high_res is a tensor of shape (batch_size, 3, base_x*4, base_y*4), normalized to [-1, 1]
            device: The device to use for getting model outputs and computing loss.
        Returns:
            The loss.
    """
    tokens, masks, low_res_image, high_res_image = [ x.to(device) for x in batch ]
    timesteps = th.randint(0, len(glide_diffusion.betas) - 1, (low_res_image.shape[0],), device=device)
    noise = th.randn_like(high_res_image, device=device) # Noise should be shape of output i think
    noised_high_res_image = glide_diffusion.q_sample(high_res_image, timesteps, noise=noise).to(device)
    _, C = noised_high_res_image.shape[:2]
    model_output = glide_model(
        noised_high_res_image.to(device),
        timesteps.to(device),
        low_res=low_res_image.to(device),
        tokens=tokens.to(device),
        mask=masks.to(device))
    epsilon, _ = th.split(model_output, C, dim=1)
    return th.nn.functional.mse_loss(epsilon, noise.to(device).detach())


def run_glide_finetune_epoch(
    glide_model: Text2ImUNet,
    glide_diffusion: SpacedDiffusion,
    glide_options: dict,
    D_model: Discriminator, # Imported from train_glide_adversarial
    dataloader: th.utils.data.DataLoader,
    optimizer: th.optim.Optimizer,
    D_optimizer: th.optim.Optimizer,
    D_criterion: th.nn.BCELoss, # idk if this will work
    sample_bs: int,  # batch size for inference
    sample_gs: float = 4.0,  # guidance scale for inference
    sample_respacing: str = '100', # respacing for inference
    prompt: str = "",  # prompt for inference, not training
    side_x: int = 64,
    side_y: int = 64,
    outputs_dir: str = "./outputs",
    checkpoints_dir: str = "./finetune_checkpoints",
    device: str = "cpu",
    log_frequency: int = 100,
    wandb_run=None,
    gradient_accumualation_steps=1,
    epoch: int = 0,
    train_upsample: bool = False,
    upsample_factor=4,
    image_to_upsample='low_res_face.png',
):
    if train_upsample: train_step = upsample_train_step
    else: train_step = base_train_step

    os.makedirs(checkpoints_dir, exist_ok=True)
    glide_model.to(device)
    glide_model.train()
    log = {}
    D_model.to(device) # configuring discriminator
    D_model.train()
    for train_idx, batch in tqdm(enumerate(dataloader)):
        accumulated_loss, fakes, reals = train_step(
            glide_model=glide_model,
            glide_diffusion=glide_diffusion,
            batch=batch,
            device=device,
        )

        # Discriminator step
        # see https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
        #torch.full((reals.size,), real_label, dtype=torch.float, device=device)
        real_preds = D_model(reals.to(device).detach())
        real_labels = th.ones_like(real_preds)
        D_real_loss = D_criterion(real_preds, real_labels)

         #torch.full((fakes.size,), fake_label, dtype=torch.float, device=device)
        fake_preds = D_model(fakes.to(device).detach())
        fake_labels = th.zeros_like(fake_preds)
        D_fake_loss = D_criterion(fake_preds, fake_labels)

        D_loss = D_real_loss + D_fake_loss
        D_loss.backward(retain_graph=True)

        print("D_LOSS: ", D_loss)

        print("GLIDE_LOSS: ", accumulated_loss)

        # Update GLIDE
        disc_glide_loss = 0.10*D_criterion(fake_preds, real_labels)
        plain_glide_loss = accumulated_loss.item()
        accumulated_loss += disc_glide_loss
        print("GLIDE_D_LOSS: ", disc_glide_loss)
        accumulated_loss.backward()

        # Steps for both models
        # GLIDE
        optimizer.step()
        glide_model.zero_grad()
        # Discriminator
        D_optimizer.step()
        D_model.zero_grad()


        print("ITER: ",train_idx)
        # print("Loss: ", acccumulated_loss.item())
        log = {**log, "iter": train_idx, "GLIDE Loss":plain_glide_loss, \
        "GLIDE + D Loss": accumulated_loss.item() / gradient_accumualation_steps, \
        "D Loss":disc_glide_loss}
        tqdm.write(f"loss: {accumulated_loss.item():.4f}")
        # Sample from the model
        if train_idx > 0 and train_idx % log_frequency == 0:
            tqdm.write(f"Sampling from model at iteration {train_idx}")
            samples = glide_util.sample(
                glide_model=glide_model,
                glide_options=glide_options,
                side_x=side_x,
                side_y=side_y,
                prompt=prompt,
                batch_size=sample_bs,
                guidance_scale=sample_gs,
                device=device,
                prediction_respacing=sample_respacing,
                #upsample_factor=upsample_factor,
                image_to_upsample=image_to_upsample,
            )
            sample_save_path = os.path.join(outputs_dir, f"{train_idx}.png")
            train_util.pred_to_pil(samples).save(sample_save_path)
            wandb_run.log(
                {
                    **log,
                    "iter": train_idx,
                    "samples": wandb.Image(sample_save_path, caption=prompt),
                }
            )
            tqdm.write(f"Saved sample {sample_save_path}")
        if train_idx % 5000 == 0 and train_idx > 0:
            train_util.save_model(glide_model, checkpoints_dir, train_idx, epoch)
            tqdm.write(
                f"Saved checkpoint {train_idx} to {checkpoints_dir}/glide-ft-{train_idx}.pt"
            )
        if train_idx > 8770: # for 00011.json
            train_util.save_model(glide_model, checkpoints_dir, train_idx, epoch)
            tqdm.write(
                f"Saved checkpoint {train_idx} to {checkpoints_dir}/glide-ft-{train_idx}.pt"
            )
        wandb_run.log(log)
    tqdm.write(f"Finished training, saving final checkpoint")
    train_util.save_model(glide_model, checkpoints_dir, train_idx, epoch)
