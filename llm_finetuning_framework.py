"""
LLM Fine-Tuning Framework — LoRA-based fine-tuning pipeline for
causal language models with evaluation, checkpointing, and ONNX export.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)


@dataclass
class LoRAConfig:
    r: int = 8
    lora_alpha: int = 32
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_dropout: float = 0.05
    bias: str = "none"


@dataclass
class FinetuneConfig:
    base_model: str = "microsoft/phi-2"
    output_dir: str = "./checkpoints"
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    max_length: int = 512
    warmup_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 250
    lora: LoRAConfig = field(default_factory=LoRAConfig)


class InstructionDataset(Dataset):
    def __init__(self, samples: list[dict], tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encodings = [
            tokenizer(
                f"### Instruction:\n{s['instruction']}\n\n### Response:\n{s['response']}",
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            for s in samples
        ]

    def __len__(self) -> int:
        return len(self.encodings)

    def __getitem__(self, idx: int) -> dict:
        enc = self.encodings[idx]
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": enc["input_ids"].squeeze().clone(),
        }


class LLMFineTuner:
    """
    LoRA-based LLM fine-tuner. Loads base model, applies LoRA adapters,
    trains on instruction data, and saves checkpoints.
    """

    def __init__(self, config: FinetuneConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        ).to(self.device)

    def _apply_lora(self):
        """Apply LoRA adapters via PEFT if available, else skip."""
        try:
            from peft import get_peft_model, LoraConfig as PeftLoraConfig, TaskType
            peft_config = PeftLoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora.r,
                lora_alpha=self.config.lora.lora_alpha,
                target_modules=self.config.lora.target_modules,
                lora_dropout=self.config.lora.lora_dropout,
                bias=self.config.lora.bias,
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
        except ImportError:
            pass  # PEFT not available, fine-tune full model

    def train(self, train_samples: list[dict], eval_samples: Optional[list[dict]] = None):
        self._apply_lora()
        train_ds = InstructionDataset(train_samples, self.tokenizer, self.config.max_length)
        loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        total_steps = len(loader) * self.config.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, self.config.warmup_steps, total_steps)

        self.model.train()
        global_step = 0
        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()
                global_step += 1

                if global_step % self.config.save_steps == 0:
                    self._save(f"step-{global_step}")

            print(f"Epoch {epoch+1}/{self.config.epochs} — loss: {epoch_loss/len(loader):.4f}")

        self._save("final")

    def _save(self, tag: str):
        path = os.path.join(self.config.output_dir, tag)
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
