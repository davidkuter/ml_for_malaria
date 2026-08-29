from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml_for_malaria.model.predict import prepare_predict_smiles
from ml_for_malaria.runs.checkpoints import RunCheckpointer
from ml_for_malaria.schemas import (
    Architecture,
    ModelMeta,
    Predictions,
    PretrainedCheckpoint,
)

ARCHITECTURE = Architecture.CHEMBERTA
DEFAULT_PRETRAINED_NAME = PretrainedCheckpoint.CHEMBERTA_77M_MTR
TINY_TEST_NAME = PretrainedCheckpoint.TINY_TEST
MAX_LENGTH = 128
_TINY_HIDDEN = 16
_TINY_LAYERS = 1
_TINY_HEADS = 2
_TINY_INTERMEDIATE = 32
_TINY_MAX_POSITIONS = 256


def _require_transformers():
    try:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            PreTrainedTokenizerFast,
            RobertaConfig,
            RobertaForSequenceClassification,
        )
    except ImportError as exc:
        raise ImportError(
            "SMILES transformers require the optional 'dl' extra "
            "(torch, transformers). Install with: uv sync --extra dl"
        ) from exc
    return (
        torch,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        PreTrainedTokenizerFast,
        RobertaConfig,
        RobertaForSequenceClassification,
    )


def tiny_smiles_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    specials = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    alphabet = list("()[]+#-=\\/@%.CNOSPFBHIcnosp0123456789=*")
    vocab = {tok: i for i, tok in enumerate(specials + alphabet)}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Split(pattern=".", behavior="isolated")
    bos_id = vocab["<s>"]
    eos_id = vocab["</s>"]
    backend.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[
            ("<s>", bos_id),
            ("</s>", eos_id),
        ],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        cls_token="<s>",
        sep_token="</s>",
        mask_token="<mask>",
        model_max_length=MAX_LENGTH,
    )


def load_tokenizer_and_model(
    pretrained_name: str,
    num_labels: int = 2,
    freeze_encoder: bool = True,
):
    (
        _,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        _,
        RobertaConfig,
        RobertaForSequenceClassification,
    ) = _require_transformers()
    if pretrained_name == TINY_TEST_NAME:
        tokenizer = tiny_smiles_tokenizer()
        config = RobertaConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=_TINY_HIDDEN,
            num_hidden_layers=_TINY_LAYERS,
            num_attention_heads=_TINY_HEADS,
            intermediate_size=_TINY_INTERMEDIATE,
            max_position_embeddings=_TINY_MAX_POSITIONS,
            num_labels=num_labels,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            type_vocab_size=1,
        )
        model = RobertaForSequenceClassification(config)
    else:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_name, num_labels=num_labels
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.pad_token_id
    if freeze_encoder:
        _freeze_encoder(model)
    return tokenizer, model


def _freeze_encoder(model) -> None:
    encoder = getattr(model, "roberta", None) or getattr(model, "bert", None)
    if encoder is not None:
        for param in encoder.parameters():
            param.requires_grad = False
        return
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False


def _positive_class_proba(logits) -> np.ndarray:
    torch, *_ = _require_transformers()
    probs = torch.softmax(logits, dim=-1)
    if probs.shape[-1] == 1:
        return probs.detach().cpu().numpy().reshape(-1)
    return probs[:, 1].detach().cpu().numpy()


class SmilesTransformerClassifier:
    """Hugging Face SMILES transformer restored from a training run directory."""

    architecture = ARCHITECTURE

    def __init__(self, model, tokenizer, metadata: ModelMeta):
        self.model = model
        self.tokenizer = tokenizer
        self.metadata = metadata

    @classmethod
    def load(cls, outdir: str | Path) -> SmilesTransformerClassifier:
        _, AutoModelForSequenceClassification, AutoTokenizer, *_ = (
            _require_transformers()
        )
        ckpt = RunCheckpointer(outdir)
        config_path = ckpt.hf_model_dir / "config.json"
        if not ckpt.meta_path.exists() or not config_path.exists():
            raise FileNotFoundError(
                f"No saved transformer in {ckpt.outdir}. Expected hf_model/ and model_meta.json"
            )
        metadata = ModelMeta.model_validate(ckpt.load_json(ckpt.meta_path))
        if metadata.architecture != ARCHITECTURE:
            raise ValueError(
                f"Run directory {ckpt.outdir} was trained with "
                f"architecture={metadata.architecture!r}, "
                f"not {ARCHITECTURE!r}."
            )
        try:
            tokenizer = AutoTokenizer.from_pretrained(ckpt.hf_model_dir)
        except (OSError, ValueError, AttributeError):
            if metadata.pretrained_name == TINY_TEST_NAME:
                tokenizer = tiny_smiles_tokenizer()
            else:
                raise
        model = AutoModelForSequenceClassification.from_pretrained(ckpt.hf_model_dir)
        model.eval()
        return cls(model=model, tokenizer=tokenizer, metadata=metadata)

    def predict(self, smiles: list[str] | pd.Series) -> pd.DataFrame:
        torch, *_ = _require_transformers()
        output, sanitized, unique = prepare_predict_smiles(smiles)
        probability = Predictions.PROBABILITY
        if unique.empty:
            return Predictions.validate(output.assign(**{probability: np.nan}))

        encoded = self.tokenizer(
            list(unique),
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        self.model.eval()
        with torch.no_grad():
            logits = self.model(**encoded).logits
        mapped = pd.Series(
            _positive_class_proba(logits),
            index=list(unique),
            name=probability,
        )
        output[probability] = sanitized.map(mapped)
        return Predictions.validate(output)
