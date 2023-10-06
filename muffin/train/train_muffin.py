# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import timm
import json
import torch
import logging
import pathlib
import getpass
import transformers
from PIL import Image

from typing import Dict, Optional, Sequence, List
from dataclasses import dataclass, field
from torch.utils.data import Dataset

from utils.utils import is_main_process, get_rank
from muffin.train.trainers import MuffinTrainer
from muffin import conversation as conversation_lib
from muffin import LlavaLlamaForCausalLM, Beit3LlavaLlamaForCausalLM
from muffin.model.muffin import interpolate_beit3
from muffin.model.utils import stop_gradient_by_name
from muffin.train.train_utils import SFT_collator_fn, IGNORE_INDEX, encode_multimodal_sample, encode_multimodal_preference_sample

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_use_im_start_end: bool = field(default=False)
    num_query: int = 64


@dataclass
class DataArguments:
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_token_len: int = 0
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    parquet: bool = False
    data_source_names: str = 'unimm-chat'
    data_source_weights: str = '100'


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    force_fsdp: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    max_steps: int = field(default=1_000)
    no_randaug: bool = False
    fully_tune: bool = False


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


class LazySupervisedDataset(Dataset):
    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizer,
                 multimodal_cfg: dict):
        super(LazySupervisedDataset, self).__init__()

        logging.warning("Loading data...")
        list_data_dict = [json.loads(line) for line in open('./data/' + multimodal_cfg['data_source_names'][0] + '.json')]

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.multimodal_cfg = multimodal_cfg

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        source: dict = self.list_data_dict[i]

        img_path = self.multimodal_cfg['image_folder'] + source['image_name'] + '.jpg'
        source['image'] = Image.open(img_path).convert('RGB')
        source['conversations'] = source['conversation']

        data_dict = encode_multimodal_sample(source, self.tokenizer, self.multimodal_cfg)
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        return SFT_collator_fn(instances, self.tokenizer.pad_token_id)

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                multimodal_cfg=dict(
                                    is_multimodal=data_args.is_multimodal,
                                    image_token_len=data_args.image_token_len,
                                    image_folder=data_args.image_folder,
                                    image_aspect_ratio=data_args.image_aspect_ratio,
                                    use_im_start_end=getattr(data_args, 'mm_use_im_start_end', False),
                                    image_processor=getattr(data_args, 'train_image_processor', None),
                                    data_source_names=getattr(data_args, 'data_source_names'),
                                    data_source_weights=getattr(data_args, 'data_source_weights')))
    print(f'Train data size is {len(train_dataset)}', flush=True)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def init_model(model_args, data_args, training_args):
    if model_args.vision_tower is not None:
        model = Beit3LlavaLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir
        )
    model.config.use_cache = False

    if (hasattr(model.config, "mm_vision_tower") and
        model_args.vision_tower is not None and
        model_args.vision_tower != model.config.mm_vision_tower):

        print(f'Update vision arch from {model.config.mm_vision_tower} to {model_args.vision_tower}', flush=True)
        model.config.mm_vision_tower = model_args.vision_tower

        # may interpolate
        state_dict = interpolate_beit3(model.model.vision_tower, model_args.vision_tower)
        new_vision_tower = timm.create_model(model_args.vision_tower)
        new_vision_tower.load_state_dict(state_dict)
        model.model.vision_tower = new_vision_tower

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=tokenizer,
                model=model,
            )
        if "llama" in model_args.model_name_or_path:
            tokenizer.add_special_tokens({
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            })
    else:
        tokenizer.pad_token = tokenizer.unk_token
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1_1"]

    if model_args.vision_tower is not None:
        model_vision_dict = model.model.initialize_vision_modules(
            vision_tower=model_args.vision_tower,
            no_randaug=training_args.no_randaug,
            num_query=model_args.num_query,
        )
        dtype = torch.float32
        if training_args.fp16:
            dtype = torch.float16
        if training_args.bf16:
            dtype = torch.bfloat16
        if training_args.fully_tune:
            dtype = torch.float32
        model.model.vision_tower.to(dtype=dtype, device=training_args.device)
        vision_config = model_vision_dict['vision_config']

        data_args.image_token_len = model_vision_dict['image_token_len']
        data_args.train_image_processor, data_args.test_image_processor = model_vision_dict['image_processor']
        data_args.is_multimodal = True

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        model.config.fully_tune = training_args.fully_tune
        assert model_args.tune_mm_mlp_adapter ^ model.config.fully_tune, f'Value of fully_tune and tune_mm_mlp_adapter are same: {model_args.tune_mm_mlp_adapter} {model.config.fully_tune}'
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.model.mm_projector.parameters():
                p.requires_grad = True

            model.model.query.requires_grad_(True)
            beit3 = model.model.vision_tower.beit3
            beit3.requires_grad_(True)
            if training_args.deepspeed:
                beit3.vision_embed.requires_grad_(False)
                beit3.apply(stop_gradient_by_name('A'))
            else:
                # with torch DDP, set zero LR rather than stop gradient
                # to accelerate training
                pass
        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.model.mm_projector.parameters():
                p.requires_grad = False

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end

        vision_config.use_im_start_end = training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.initialize_vision_tokenizer(mm_use_im_start_end=model_args.mm_use_im_start_end, tokenizer=tokenizer, device=training_args.device,
                                          tune_mm_mlp_adapter=model_args.tune_mm_mlp_adapter)

        if model.config.fully_tune:
            model.requires_grad_(True)

        # remove unused params
        model.model.vision_tower.beit3.vision_embed.mask_token = None
        model.model.vision_tower.beit3.text_embed = None

        params_no_grad = [n for n, p in model.named_parameters() if not p.requires_grad]
        if is_main_process():
            print(f'No grad params are : {params_no_grad}', flush=True)
        if len(params_no_grad) > 0:
            if training_args.fsdp is not None and len(training_args.fsdp) > 0:
                if len(params_no_grad) < 10:
                    print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}'. format(len(params_no_grad), params_no_grad))
                else:
                    print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}...(omitted)'. format(len(params_no_grad), ', '.join(params_no_grad[:10])))
                print("[WARNING] Attempting to use FSDP with partially frozen paramters, this is experimental.")
                print("[WARNING] As of 4/30/23, this feature requires PyTorch-nightly build.  See here for details: https://github.com/haotian-liu/LLaVA#experimental-use-fsdp-to-save-memory-in-pretraining")

                from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
                def patch_FSDP_use_orig_params(func):
                    def wrap_func(*args, **kwargs):
                        use_orig_params = kwargs.pop('use_orig_params', True)
                        return func(*args, **kwargs, use_orig_params=use_orig_params)
                    return wrap_func

                FSDP.__init__ = patch_FSDP_use_orig_params(FSDP.__init__)

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    return model.cuda(), data_module, tokenizer


def get_local_dir(prefixes_to_resolve: List[str]) -> str:
    """Return the path to the cache directory for this user."""
    for prefix in prefixes_to_resolve:
        if os.path.exists(prefix):
            return f"{prefix}/{getpass.getuser()}"
    os.makedirs(prefix)
    return f"{prefix}/{getpass.getuser()}"


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.report_to == 'wandb':
        os.environ['WANDB_CACHE_DIR'] = get_local_dir(['.cache', '_temp'])

    data_args.data_source_names = data_args.data_source_names.split('#')
    data_args.data_source_weights = [int(x) for x in data_args.data_source_weights.split('#')]

    model, data_module, tokenizer = init_model(model_args, data_args, training_args)

    trainer = MuffinTrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)

    # print(f'Training args: {training_args}')
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        print(f'Resume from checkpoint.')
        trainer.train(resume_from_checkpoint=True)
    else:
        print(f'Train from start.')
        trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer,
                                   output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
