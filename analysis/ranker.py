"""Shared GuideRanker model, loading, and tokenizer utilities."""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast


class GuideRanker(nn.Module):
    def __init__(self, model_name: str, num_classes: int):
        super().__init__()
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )
        hidden_size = self.backbone.config.hidden_size
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )
        self.head.float()

    def unfreeze_last_n_layers(self, n: int = 2) -> None:
        for layer in self.backbone.model.layers[-n:]:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        attention_mask = attention_mask.to(torch.long)
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1]
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        pooled = hidden[batch_idx, seq_lengths]
        return self.head(pooled.float())


def format_input(goal: str, guide: str) -> str:
    return f"Goal: {goal}\nGuide: {guide}"


def load_tokenizer(model_name: str) -> PreTrainedTokenizerFast:  # pyright: ignore[reportReturnType]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_checkpoint(
    checkpoint_path: Path, device: torch.device
) -> tuple[GuideRanker, dict, PreTrainedTokenizerFast]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = GuideRanker(config["model_name"], config["num_classes"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    tokenizer = load_tokenizer(config["model_name"])
    return model, config, tokenizer


def find_latest_checkpoint() -> Path:
    runs_dir = Path(__file__).parent / "runs"
    candidates = sorted(
        runs_dir.glob("ranker-*/ranker.pt"), key=lambda p: p.stat().st_mtime
    )
    if not candidates:
        print(f"No checkpoints found in {runs_dir.resolve()}", file=sys.stderr)
        sys.exit(1)
    return candidates[-1]
