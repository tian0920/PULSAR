import copy
import logging
import torch
from data.utils import IGNORE_INDEX
from trainer.utils import compute_kl_divergence
from trainer.unlearn.base import UnlearnTrainer

logger = logging.getLogger(__name__)


class GradDiff(UnlearnTrainer):
    def __init__(self, gamma=1.0, alpha=1.0, retain_loss_type="NLL", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.retain_loss_type = retain_loss_type
        self.ref_model = None
        if retain_loss_type == "KL":
            self.ref_model = self._prepare_ref_model(self.model)

    def _sanitize_inputs(self, inputs):
        """
        Make sure tensor shapes/dtypes are consistent and token ids stay in range.
        This prevents CUDA device-side asserts when the model receives malformed inputs.
        """
        if "input_ids" in inputs:
            input_ids = inputs["input_ids"]
            if input_ids.dtype != torch.long:
                input_ids = input_ids.long()
            seq_len = input_ids.shape[-1]

            attn = inputs.get("attention_mask", None)
            if attn is not None and attn.shape[-1] != seq_len:
                if attn.shape[-1] > seq_len:
                    attn = attn[..., :seq_len]
                else:
                    pad = torch.zeros(
                        attn.shape[:-1] + (seq_len - attn.shape[-1],),
                        dtype=attn.dtype,
                        device=attn.device,
                    )
                    attn = torch.cat([attn, pad], dim=-1)
                inputs["attention_mask"] = attn

            vocab_size = getattr(self.model.config, "vocab_size", None)
            if vocab_size is not None:
                bad = (input_ids < 0) | (input_ids >= vocab_size)
                if bad.any():
                    pad_id = getattr(self.tokenizer, "pad_token_id", None)
                    fill_val = pad_id if pad_id is not None else vocab_size - 1
                    input_ids = input_ids.clone()
                    input_ids[bad] = fill_val
                    logger.warning(
                        "Found %d out-of-range input_ids; clamped to %s",
                        bad.sum().item(),
                        fill_val,
                    )
            inputs["input_ids"] = input_ids

        if "labels" in inputs:
            labels = inputs["labels"]
            vocab_size = getattr(self.model.config, "vocab_size", None)
            if vocab_size is not None:
                bad = (labels != IGNORE_INDEX) & (
                    (labels < 0) | (labels >= vocab_size)
                )
                if bad.any():
                    labels = labels.clone()
                    labels[bad] = IGNORE_INDEX
                    logger.warning(
                        "Found %d out-of-range labels; set to IGNORE_INDEX",
                        bad.sum().item(),
                    )
            inputs["labels"] = labels

        return inputs

    def _prepare_ref_model(self, model):
        ref_model = copy.deepcopy(model).to(self.accelerator.device)
        ref_model.eval()
        if self.is_deepspeed_enabled:
            ref_model = self._prepare_deepspeed(ref_model)
        else:
            ref_model = self.accelerator.prepare_model(ref_model, evaluation_mode=True)
        return ref_model

    def compute_retain_loss(self, model, retain_inputs):
        retain_inputs = self._sanitize_inputs(retain_inputs)
        retain_outputs = model(**retain_inputs)
        retain_loss = 0.0
        if self.retain_loss_type == "NLL":
            retain_loss += retain_outputs.loss
        elif self.retain_loss_type == "KL":
            kl_loss, retain_outputs = compute_kl_divergence(
                self.model, self.ref_model, retain_inputs
            )
            retain_loss += kl_loss
        else:
            raise NotImplementedError(
                f"{self.retain_loss_type} not implemented for retain set"
            )
        return retain_loss

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):

        forget = inputs["forget"]
        forget = forget["forget"]
        forget_inputs = {
            "input_ids": forget["input_ids"],
            "attention_mask": forget["attention_mask"],
            "labels": forget["labels"],
        }
        forget_inputs = self._sanitize_inputs(forget_inputs)

        forget_outputs = model(**forget_inputs)
        forget_loss = -forget_outputs.loss

        retain_loss = 0.0
        # Skip the retain forward pass entirely when alpha == 0 to reduce memory (e.g., revocation stage)
        if self.alpha != 0:
            retain = inputs["retain"]
            retain = retain["retain"]
            retain_inputs = {
                "input_ids": retain["input_ids"],
                "attention_mask": retain["attention_mask"],
                "labels": retain["labels"],
            }
            retain_inputs = self._sanitize_inputs(retain_inputs)
            retain_loss = self.compute_retain_loss(
                model=model, retain_inputs=retain_inputs
            )

        loss = self.gamma * forget_loss + self.alpha * retain_loss

        return (loss, forget_outputs) if return_outputs else loss
