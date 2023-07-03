import json
import os
import sys
import ray
from ray.train.huggingface.accelerate import AccelerateTrainer
from ray.air import ScalingConfig

from datasets import load_dataset

import trlx
from trlx.data.default_configs import (
    ILQLConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)

config = TRLConfig(
    train=TrainConfig(
        seq_length=256,
        batch_size=1,
        epochs=100,
        total_steps=int(1e10),
        checkpoint_interval=10000,
        eval_interval=1000,
        pipeline="PromptPipeline",
        trainer="AccelerateILQLTrainer",
        checkpoint_dir="checkpoints/ilql_hh",
    ),
    model=ModelConfig(model_path="EleutherAI/gpt-j-6B", num_layers_unfrozen=2),
    tokenizer=TokenizerConfig(tokenizer_path="EleutherAI/gpt-j-6B", truncation_side="left"),
    optimizer=OptimizerConfig(name="adamw", kwargs=dict(lr=1e-6, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)),
    scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=1000000000, eta_min=1e-6)),
    method=ILQLConfig(
        name="ilqlconfig",
        tau=0.6,
        gamma=0.99,
        cql_scale=0.1,
        awac_scale=1,
        alpha=0.0001,
        beta=0,
        steps_for_target_q_sync=1,
        two_qs=True,
        gen_kwargs=dict(max_new_tokens=128, top_k=20, beta=1, temperature=1.0),
    ),
)

def preprocess(sample):
    sample["prompt_output"] = [
        [sample["prompt"], sample["chosen"]],
        [sample["prompt"], sample["rejected"]],
    ]
    sample["reward"] = [1, -1]
    return sample


def main():
    dataset = load_dataset("Dahoas/static-hh").map(preprocess)
    prompts_outputs = sum(dataset["train"]["prompt_output"], [])
    eval_prompts = dataset["test"]["prompt"][:128]
    rewards = sum(dataset["train"]["reward"], [])

    trlx.train(
        samples=prompts_outputs,
        rewards=rewards,
        config=config,
        eval_prompts=eval_prompts,
        stop_sequences=["Human:", "human:", "Assistant:", "assistant:"],
    )

if __name__ == '__main__':
    ray.init()
    AccelerateTrainer(
        main,
        accelerate_config="configs/accelerate/deepspeed_json.yaml",
        scaling_config=ScalingConfig(
            trainer_resources={"CPU": 0},
            num_workers=16,
            use_gpu=True,
            resources_per_worker={"CPU": 12, "GPU": 1}
        )
    ).fit()
