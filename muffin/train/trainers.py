from torch import nn
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names
from transformers.utils.import_utils import is_sagemaker_mp_enabled
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
import os
import torch
import wandb
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from transformers import Trainer
from typing import Any, Dict, Optional, Tuple, Union
from torch import Tensor
from torch.nn import Module
from utils.utils import is_main_process


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


class LLaVATrainer(Trainer):

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            # Save the model
            _state_dict = state_dict
            if _state_dict is None:
                # Only save the model itself if we are using distributed training
                model_to_save = unwrap_model(self.model)
                _state_dict = model_to_save.state_dict()

            weight_to_save = {}
            keys_to_match = ['mm_projector', 'embed_tokens', 'embed_in']
            for k, v in _state_dict.items():
                if any(key_match in k for key_match in keys_to_match):
                    weight_to_save[k] = v

            current_folder = output_dir.split('/')[-1]
            parent_folder = os.path.dirname(output_dir)
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(
                    mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))

        super(LLaVATrainer, self)._save(output_dir, state_dict)

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        return None

    def get_train_dataloader(self) -> DataLoader:
        print(f'Start creating loader')
        dataloader_train = DataLoader(
            self.train_dataset,
            sampler=None,
            batch_size=self._train_batch_size,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.data_collator
        )
        print(f'End creating loader')
        return dataloader_train


class MuffinTrainer(Trainer):

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]

        def should_zero_lr(param_name: str):
            if 'beit3' in param_name:
                if '.A' in param_name:
                    return True
                if 'beit3.vision_embed' in param_name:
                    return True
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters() if (p.requires_grad and should_zero_lr(n))
                ],
                "weight_decay": self.args.weight_decay,
                "lr": 0.0,
                "initial_lr": 0.0
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad and not should_zero_lr(n))
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad and not should_zero_lr(n))
                ],
                "weight_decay": 0.0,
            },
        ]
        for n, p in model.named_parameters():
            # print(f'Check LR of {n}')
            if should_zero_lr(n) and is_main_process():
                print(f'Zero LR params: {n}', flush=True)

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        self.scheduler = self.create_scheduler(
            num_training_steps=num_training_steps, optimizer=self.optimizer)
        print(f'LR schduler is ', self.scheduler)

