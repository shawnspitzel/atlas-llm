import torch
import torch.distributed as dist
from collections.abc import Iterable


class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, optimizer, **kwargs):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.params = []
        self.optimizer_cls = optimizer
        self.kwargs = kwargs
        self.optimizer = None

        if not isinstance(params, (list, tuple)):
            params = list(params)

        super().__init__(params, defaults={})

    def extract_params(self, params):
        if isinstance(params, dict):
            return list(params["params"])
        elif isinstance(params, Iterable):
            return list(params)
        else:
            raise ValueError("Invalid data-type for params")

    def shard_params(self, params, start_idx):
        local_params = []
        for i, param in enumerate(params):
            if (start_idx + i) % self.world_size == self.rank:
                local_params.append(param)
        return local_params

    def sync_params(self):
        for i, param in enumerate(self.params):
            dist.broadcast(param.data, src=i % self.world_size)

    def step(self, closure=None, **kwargs):
        loss = None
        if self.optimizer is not None:
            loss = self.optimizer.step(closure, **kwargs)
        self.sync_params()
        return loss

    def zero_grad(self, set_to_none: bool = True):
        for param in self.params:
            if param.grad is not None:
                if set_to_none:
                    param.grad = None
                else:
                    param.grad.zero_()

    def add_param_group(self, param_group):
        params = self.extract_params(param_group)
        start_idx = len(self.params)
        self.params.extend(params)
        local_params = self.shard_params(params, start_idx)

        if local_params:
            if self.optimizer is None:
                self.optimizer = self.optimizer_cls(local_params, **self.kwargs)
            else:
                local_param_group = {**param_group, 'params': local_params}
                self.optimizer.add_param_group(local_param_group)
