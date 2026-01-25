import torch
import torch.nn as nn
import torch.distributed as dist

class DDPIndividualParameters(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.grad_handles = []

        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        for param in self.module.parameters():
            if param.requires_grad:
                param.register_hook(self._create_grad_hook(param))

    def _create_grad_hook(self, grad):
        # async_op=True returns immediately with a handle to overlap communication
        handle = dist.all_reduce(grad.data, op=dist.ReduceOp.SUM, async_op=True)
        self.grad_handles.append(handle)
        return grad

    def finish_gradient_synchronization(self):
        for handle in self.grad_handles:
            handle.wait()

        # all_reduce sums gradients, so divide by world_size to get average
        world_size = dist.get_world_size()
        for param in self.module.parameters():
            if param.requires_grad and param.grad is not None:
                param.grad.data.div_(world_size)

        self.grad_handles = []

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def get_ddp_individual_parameters(module: torch.nn.Module) -> torch.nn.Module:
    return DDPIndividualParameters(module)


def ddp_individual_parameters_on_after_backward(ddp_model: torch.nn.Module):
    ddp_model.finish_gradient_synchronization()
