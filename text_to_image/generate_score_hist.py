import sys
sys.path.append('fkd_diffusers')

import os
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import numpy as np
from copy import deepcopy

import torch

from fks_utils import get_model, do_eval


args = dict(
    output_dir="samples_for_paper",
    eta=1.0,
    guidance_reward_fn="ImageReward",
    metrics_to_compute="ImageReward#HumanPreference",
    seed=42,
)

args = argparse.Namespace(**args)
print(args)

# # cache metric fns
# do_eval(
#     prompt=["test"],
#     images=[Image.new("RGB", (224, 224))],
#     metrics_to_compute=["ImageReward", "HumanPreference"],
# )


# seed everything
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


def generate_config(start_seed=42, num_seeds=5):
    base_fkd_args = dict(
        lmbda=2.0,
        use_smc=True,
        adaptive_resampling=True,
        resample_frequency=20,
        resampling_t_start=20,
        resampling_t_end=80,
        guidance_reward_fn="ImageReward",
        metric_to_chase=None, # should be specified when using "LLMGrader".
    )

    arr_fkd_args = []

    for time_steps in [100]:
        for lmbda in [10.0]:
            for num_particles in [2, 4]:
                for seed in range(start_seed, start_seed + num_seeds):
                    base_fkd_args["time_steps"] = time_steps
                    base_fkd_args["lmbda"] = lmbda
                    base_fkd_args["num_particles"] = num_particles
                    base_fkd_args["seed"] = seed
                    arr_fkd_args.append(base_fkd_args.copy())

    return arr_fkd_args


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def generate_and_save_image(images, image_fpath, num_particles):
    if num_particles > 1:
        fig, ax = plt.subplots(
            1, num_particles, figsize=(num_particles * 5, 5), dpi=200
        )

        for i, image in enumerate(images):
            ax[i].imshow(image)
            ax[i].axis("off")

        fig.tight_layout()
        plt.savefig(image_fpath)
        plt.show()
        plt.close()
    else:
        plt.imshow(images[0])
        plt.axis("off")
        plt.savefig(image_fpath)
        plt.show()
        plt.close()


# per seed per num_particles generate samples
def generate_samples(fkd_args, pipeline, prompt_data):
    all_metrics = []
    for prompt_idx, item in enumerate(prompt_data):
        prompt = [item["prompt"]] * fkd_args["num_particles"]
        prompt_ = item["prompt"].replace(" ", "_")
        lmbda_ = fkd_args["lmbda"]
        num_particles = fkd_args["num_particles"]
        time_steps_ = fkd_args["time_steps"]
        curr_seed = fkd_args["seed"]

        # Make directories for each potential type
        max_dir = os.path.join(images_fpath, "max", prompt_, str(num_particles))
        os.makedirs(max_dir, exist_ok=True)
        instant_dir = os.path.join(images_fpath, "instant", prompt_, str(num_particles))
        os.makedirs(instant_dir, exist_ok=True)
        add_dir = os.path.join(images_fpath, "add", prompt_, str(num_particles))
        os.makedirs(add_dir, exist_ok=True)
        base_dir = os.path.join(images_fpath, "base", prompt_, str(num_particles))
        os.makedirs(base_dir, exist_ok=True)

        max_fname = os.path.join(max_dir, f"seed_{curr_seed}_max.png")
        instant_fname = os.path.join(instant_dir, f"seed_{curr_seed}_instant.png")
        add_fname = os.path.join(add_dir, f"seed_{curr_seed}_add.png")
        base_fname = os.path.join(base_dir, f"seed_{curr_seed}_base.png")

        potential_to_fname = {
            "max": max_fname,
            "instant": instant_fname,
            "add": add_fname,
            "base": base_fname,
        }

        potential_types = ["max", "instant", "add", "base"]
        for potential_type in potential_types:
            # This will be the returned in the end
            eval_metrics = {}  # This will store the metrics for each run for each prompt
            eval_metrics["prompt"] = item["prompt"]  # Store the prompt
            eval_metrics["seed"] = fkd_args["seed"]
            eval_metrics["num_particles"] = fkd_args["num_particles"]

            seed_everything(0 + prompt_idx)
            fkd_type_args = deepcopy(fkd_args)
            if potential_type == "base":
                fkd_type_args["use_smc"] = False
            else:
                fkd_type_args["potential_type"] = potential_type
            
            print(f"Generating samples for {fkd_type_args}")
            images_fkd_type = pipeline(
                prompt,
                num_inference_steps=fkd_args["time_steps"],
                eta=args.eta,
                fkd_args=fkd_type_args,
            )[0]

            # Eval the results
            results = do_eval(
                            prompt=prompt,
                            images=images_fkd_type,
                            metrics_to_compute=args.metrics_to_compute.split("#"),
                        )

            # Sort images by reward
            guidance_reward = np.array(results["ImageReward"]["result"])
            sorted_idx = np.argsort(guidance_reward)[::-1]
            images_type_sorted = [images_fkd_type[i] for i in sorted_idx]
            eval_metrics["potential_type"] = potential_type  # Store the potential type
            eval_metrics["IR"] = np.array(results["ImageReward"]["result"])  # Store the results for each potential type
            eval_metrics["HPS"] = np.array(results["HumanPreference"]["result"])  # Store the results for each potential type
            generate_and_save_image(images_type_sorted, potential_to_fname[potential_type], num_particles)
            all_metrics.append(eval_metrics)

        return all_metrics


prompt_data = [
    {"prompt": "a photo of a brown knife and a blue donut"},
    {"prompt": "a photo of a blue clock and a white cup"},
    {"prompt": "a photo of an orange cow and a purple sandwich"},
    {"prompt": "a photo of a yellow bird and a black motorcycle"},
    {"prompt": "a photo of a green tennis racket and a black dog"},
    {"prompt": "a green stop sign in a red field"},
]


for model_name in [
    "stable-diffusion-v1-5",
]:
    # load model
    pipeline = get_model(model_name)

    # set output directory
    arr_fkd_args = generate_config()  # This can generate configs for multiple runs
    output_dir = os.path.join(args.output_dir)
    output_dir += f"_{args.metrics_to_compute}" 
    if arr_fkd_args[0]["metric_to_chase"]:
        output_dir += f'_{arr_fkd_args[0]["metric_to_chase"]}'
    os.makedirs(output_dir, exist_ok=True)

    images_path = output_dir + f"/{model_name}"
    os.makedirs(images_path, exist_ok=True)

    pipeline = pipeline.to("cuda")

    metrics = []

    for fkd_args in arr_fkd_args:
        print(fkd_args)
        run_metrics = generate_samples(fkd_args, pipeline, prompt_data)
        metrics.append(run_metrics)  # Get results for each run

    import pandas as pd
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(images_path, "metrics.csv"), index=False)
    print(f"Saved metrics to {os.path.join(images_path, 'metrics.csv')}")
