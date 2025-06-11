# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang)
#               2021 Hongji Wang (jijijiang77@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch


def load_checkpoint(model: torch.nn.Module, path: str):
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint, strict=False)


def save_checkpoint(model: torch.nn.Module, path: str):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(state_dict, path)

def load_init_whisper(model: torch.nn.Module, path: str):
    with (open(path, "rb")) as fp:
        checkpoint = torch.load(fp, map_location="cpu")

    if "model_state_dict" in checkpoint.keys():
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # state_dict = checkpoint["model_state_dict"]
    # state_dict = checkpoint
    # print(state_dict.keys())
    # print(checkpoint["dims"])

    updated_state_dict = {}
    for name, param in state_dict.items():
        new_name = name.replace("encoder", "whisper")
        updated_state_dict[new_name] = param

    current_model_state_dict = model.state_dict()
    loaded_list = []
    for name, param in updated_state_dict.items():
        if name in current_model_state_dict:
            print(f"Loading parameter: {name}")
            current_model_state_dict[name] = param
            loaded_list.append(name)

    for name, param in model.named_parameters():
        if name in loaded_list:
            param.requires_grad = False

    model.load_state_dict(current_model_state_dict, strict=False)
