import os
import typing
import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from src.training.utils.storage import S3Storage

def data_loading(x: npt.NDArray, batch_size: int, context_length: int, device: torch.device) -> tuple[Tensor, Tensor]:
    inputs = torch.empty((batch_size, context_length), dtype=torch.long, device=device)
    targets = torch.empty((batch_size, context_length), dtype=torch.long, device=device)
    N = len(x)
    max_start = N - context_length
    if max_start <= 0:
        raise ValueError("Dataset too small for given context_length")
    
    starts = torch.randint(
        low=0,
        high=max_start,
        size=(batch_size,),
        device=device,
    )
    inputs_np = np.empty((batch_size, context_length), dtype=x.dtype)
    targets_np = np.empty((batch_size, context_length), dtype=x.dtype)
    for i, s in enumerate(starts):
        inputs_np[i] = x[s : s + context_length]
        targets_np[i] = x[s + 1 : s + context_length + 1]
    inputs = torch.from_numpy(inputs_np).to(device, dtype=torch.long)
    targets = torch.from_numpy(targets_np).to(device, dtype=torch.long)
    return inputs, targets

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer, 
    iteration: int,
    config: dict,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
    ):
    storage_location = config.get("storage-location") if config else None
    state_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration
    }
    if storage_location == "cloud":
        S3Storage.save(state_dict, key=str(out))
    else:
        torch.save(state_dict, out)

def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict,
    device=None
    ):
    storage_location = config.get("storage-location") if config else None
    if storage_location == "cloud":
        state = S3Storage.load(key=str(src), device=device)
    else:
        if device is None:
            state = torch.load(src)
        else:
            state = torch.load(src, map_location=device)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    return state["iteration"]
    