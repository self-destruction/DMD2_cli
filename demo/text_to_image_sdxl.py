from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler, AutoencoderTiny
from main.sdxl.sdxl_text_encoder import SDXLTextEncoder
from main.utils import get_x0_from_noise
from transformers import AutoTokenizer
from accelerate import Accelerator
import random
import numpy as np
import argparse
import torch
import time
import PIL
from PIL import Image
    
SAFETY_CHECKER = False

class ModelWrapper:
    def __init__(self, args, accelerator):
        super().__init__()
        # disable all gradient calculations
        torch.set_grad_enabled(False)
        
        if args.precision == "bfloat16":
            self.DTYPE = torch.bfloat16
        elif args.precision == "float16":
            self.DTYPE = torch.float16
        else:
            self.DTYPE = torch.float32
        self.device = accelerator.device

        self.tokenizer_one = AutoTokenizer.from_pretrained(
            args.model_id, subfolder="tokenizer", revision=args.revision, use_fast=False
        )

        self.tokenizer_two = AutoTokenizer.from_pretrained(
            args.model_id, subfolder="tokenizer", revision=args.revision, use_fast=False
        )

        self.text_encoder = SDXLTextEncoder(args, accelerator, dtype=self.DTYPE)

        # vanilla SDXL VAE needs to be kept in float32
        self.vae = AutoencoderKL.from_pretrained(
            args.model_id, 
            subfolder="vae"
        ).float().to(self.device)
        self.vae_dtype = torch.float32

        self.tiny_vae = AutoencoderTiny.from_pretrained(
            "madebyollin/taesdxl", 
            torch_dtype=self.DTYPE
        ).to(self.device) 
        self.tiny_vae_dtype = self.DTYPE

        # Initialize Generator
        self.model = self.create_generator(args).to(dtype=self.DTYPE).to(self.device)

        self.accelerator = accelerator
        self.image_resolution = args.image_resolution
        self.latent_resolution = args.latent_resolution
        self.num_train_timesteps = args.num_train_timesteps
        self.vae_downsample_ratio = self.image_resolution // self.latent_resolution

        self.conditioning_timestep = args.conditioning_timestep 

        self.scheduler = DDIMScheduler.from_pretrained(
            args.model_id,
            subfolder="scheduler"
        )
        self.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)

        # sampling parameters 
        self.num_step = args.num_step 
        self.conditioning_timestep = args.conditioning_timestep 


    def create_generator(self, args):
        generator = UNet2DConditionModel.from_pretrained(
            args.model_id,
            subfolder="unet"
        ).to(self.DTYPE)

        state_dict = torch.load(args.checkpoint_path, map_location="cpu")
        # print(generator.load_state_dict(state_dict, strict=True))
        generator.requires_grad_(False)
        return generator 

    def build_condition_input(self, height, width):
        original_size = (height, width)
        target_size = (height, width)
        crop_top_left = (0, 0)

        add_time_ids = list(original_size + crop_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], device=self.device, dtype=self.DTYPE)
        return add_time_ids

    def _encode_prompt(self, prompt):
        text_input_ids_one = self.tokenizer_one(
            [prompt],
            padding="max_length",
            max_length=self.tokenizer_one.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids

        text_input_ids_two = self.tokenizer_two(
            [prompt],
            padding="max_length",
            max_length=self.tokenizer_two.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids

        prompt_dict = {
            'text_input_ids_one': text_input_ids_one.unsqueeze(0).to(self.device),
            'text_input_ids_two': text_input_ids_two.unsqueeze(0).to(self.device)
        }
        return prompt_dict 

    @staticmethod
    def _get_time():
        torch.cuda.synchronize()
        return time.time()

    def sample(
        self, noise, unet_added_conditions, prompt_embed, fast_vae_decode
    ):
        alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)

        if self.num_step == 1:
            all_timesteps = [self.conditioning_timestep]
            step_interval = 0 
        elif self.num_step == 4:
            all_timesteps = [999, 749, 499, 249]
            step_interval = 250 
        else:
            raise NotImplementedError()
        
        DTYPE = prompt_embed.dtype
        
        for constant in all_timesteps:
            current_timesteps = torch.ones(len(prompt_embed), device=self.device, dtype=torch.long)  *constant
            eval_images = self.model(
                noise, current_timesteps, prompt_embed, added_cond_kwargs=unet_added_conditions
            ).sample

            eval_images = get_x0_from_noise(
                noise, eval_images, alphas_cumprod, current_timesteps
            ).to(self.DTYPE)

            next_timestep = current_timesteps - step_interval 
            noise = self.scheduler.add_noise(
                eval_images, torch.randn_like(eval_images), next_timestep
            ).to(DTYPE)  

        if fast_vae_decode:
            eval_images = self.tiny_vae.decode(eval_images.to(self.tiny_vae_dtype) / self.tiny_vae.config.scaling_factor, return_dict=False)[0]
        else:
            eval_images = self.vae.decode(eval_images.to(self.vae_dtype) / self.vae.config.scaling_factor, return_dict=False)[0]
        eval_images = ((eval_images + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
        return eval_images 


    @torch.no_grad()
    def inference(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_images: int,
        fast_vae_decode: bool
    ):
        print("Running model inference...")

        if seed == -1:
            seed = np.random.randint(0, np.iinfo(np.int32).max)

        generator = torch.manual_seed(seed)

        add_time_ids = self.build_condition_input(height, width).repeat(num_images, 1)

        noise = torch.randn(
            num_images, 4, height // self.vae_downsample_ratio, width // self.vae_downsample_ratio, 
            generator=generator
        ).to(device=self.device, dtype=self.DTYPE) 

        prompt_inputs = self._encode_prompt(prompt)
        
        start_time = self._get_time()

        prompt_embeds, pooled_prompt_embeds = self.text_encoder(prompt_inputs)

        batch_prompt_embeds, batch_pooled_prompt_embeds = (
            prompt_embeds.repeat(num_images, 1, 1),
            pooled_prompt_embeds.repeat(num_images, 1, 1)
        )

        unet_added_conditions = {
            "time_ids": add_time_ids,
            "text_embeds": batch_pooled_prompt_embeds.squeeze(1)
        }

        eval_images = self.sample(
            noise=noise,
            unet_added_conditions=unet_added_conditions,
            prompt_embed=batch_prompt_embeds,
            fast_vae_decode=fast_vae_decode
        )

        end_time = self._get_time()

        output_image_list = [] 
        for image in eval_images:
            output_image_list.append(PIL.Image.fromarray(image.cpu().numpy()))

        return output_image_list


parser = argparse.ArgumentParser()
parser.add_argument("--latent_resolution", type=int, default=128)
parser.add_argument("--image_resolution", type=int, default=1024)
parser.add_argument("--num_train_timesteps", type=int, default=1000)
parser.add_argument("--checkpoint_path", type=str)
parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0") # RunDiffusion/Juggernaut-XL-v9
parser.add_argument("--precision", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
parser.add_argument("--conditioning_timestep", type=int, default=999)
parser.add_argument("--num_step", type=int, default=4, choices=[1, 4])
parser.add_argument("--revision", type=str)

parser.add_argument("--save_dir", type=str)
parser.add_argument("--save_file_name", type=str)

parser.add_argument("--prompt", type=str, default='An oil painting of two rabbits in the style of American Gothic, wearing the same clothes as in the original')
parser.add_argument("--seed", type=int, default=-1)
parser.add_argument("--num_images", type=int, default=1, choices=range(1, 16))
# Use Tiny VAE for faster decoding
parser.add_argument("--fast_vae_decode", action='store_true', default=False)
# Height
parser.add_argument("--h", type=int, default=1024)
# Width
parser.add_argument("--w", type=int, default=1024)

args = parser.parse_args()
print(args)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True 

accelerator = Accelerator()
model = ModelWrapper(args, accelerator)

ims = model.inference(
    args.prompt,
    args.seed,
    args.h,
    args.w,
    args.num_images,
    args.fast_vae_decode
)

for i, img in enumerate(ims):
    img.save(f'{args.save_dir}/{args.save_file_name}_{i}.png')
