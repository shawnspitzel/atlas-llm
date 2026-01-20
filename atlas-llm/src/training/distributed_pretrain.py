import os
import os.path as osp
import wandb
import numpy as np
import torch
import torch.distributed as dist
import cProfile
import pstats
import sys
import traceback
import signal
import atexit
from tqdm import tqdm
from src.training.loader import data_loading, load_checkpoint, save_checkpoint
from src.model.optimizer import AdamW, SGDOptimizer
from src.model.transformer import Transformer
from src.model.loss import cross_entropy_loss, gradient_clipping, learning_rate_schedule
from src.utils.args import get_args_pretrain
from src.systems.ddp import get_ddp_individual_parameters, ddp_individual_parameters_on_after_backward
from src.systems.optimizer_sharding import ShardedOptimizer

# Global profiler state for signal handlers
_global_profilers = {
    'cprofile': None,
    'lineprofile': None,
    'profile_dir': None,
    'base_name': None
}

def save_profile_data():
    """Save profiling data - called on exit or interruption"""
    from datetime import datetime

    if _global_profilers['cprofile'] is None:
        return

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        profile_dir = _global_profilers['profile_dir']
        base_name = _global_profilers['base_name']
        cprofile = _global_profilers['cprofile']
        lp = _global_profilers['lineprofile']

        # Save cProfile results
        cprofile_file = os.path.join(profile_dir, f"{base_name}_cprofile_{timestamp}_interrupted.txt")
        with open(cprofile_file, 'w') as f:
            stats = pstats.Stats(cprofile, stream=f)
            f.write("="*80 + "\n")
            f.write("FUNCTION-LEVEL PROFILE (cProfile) - Distributed - INTERRUPTED\n")
            f.write("="*80 + "\n\n")
            f.write("Top 30 functions by cumulative time:\n")
            f.write("-"*80 + "\n")
            stats.sort_stats('cumulative')
            stats.print_stats(30)
            f.write("\n\nTop 20 functions by total time:\n")
            f.write("-"*80 + "\n")
            stats.sort_stats('time')
            stats.print_stats(20)

        # Save line_profiler results
        lineprofile_file = os.path.join(profile_dir, f"{base_name}_lineprofile_{timestamp}_interrupted.txt")
        with open(lineprofile_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("LINE-BY-LINE PROFILE (line_profiler) - Distributed - INTERRUPTED\n")
            f.write("="*80 + "\n\n")
            lp.print_stats(stream=f)

        # Save binary cProfile data
        cprofile_binary = os.path.join(profile_dir, f"{base_name}_cprofile_{timestamp}_interrupted.prof")
        cprofile.dump_stats(cprofile_binary)

        print("\n" + "="*80)
        print("PROFILING SAVED (Distributed Training Interrupted)")
        print("="*80)
        print(f"\nProfile results saved to:")
        print(f"  1. Function-level: {cprofile_file}")
        print(f"  2. Line-by-line: {lineprofile_file}")
        print(f"  3. Binary data: {cprofile_binary}")
        print("="*80 + "\n")
    except Exception as e:
        print(f"Warning: Failed to save profiling data: {e}")

def signal_handler(signum, frame):
    """Handle Ctrl+C and other interrupts"""
    print("\n\nReceived interrupt signal, saving profiling data...")
    save_profile_data()
    sys.exit(0)


def get_checkpoint_dir(params):
    if params.get("checkpoint_dir"):
        checkpoint_dir = params["checkpoint_dir"]
    else:
        base_dir = "checkpoints"
        checkpoint_name = (
            f"nl{params['num_layers']}_"
            f"dm{params['d_model']}_"
            f"bs{params['batch_size']}_"
            f"lr{params['learning_rate']}_"
            f"seed{params['seed']}"
        )
        checkpoint_dir = osp.join(base_dir, checkpoint_name)

    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir


def pretrain(model, train_data: np.array, val_data: np.array, optimizer, params, iteration: int, use_ddp: bool = False):
    model.train()
    inputs, targets = data_loading(train_data, params["batch_size"], params["context_length"], params["device"])

    optimizer.zero_grad()
    logits = model(inputs)
    loss = cross_entropy_loss(logits, targets)
    loss.backward()

    # Synchronize gradients if using DDP
    if use_ddp:
        ddp_individual_parameters_on_after_backward(model)

    gradient_clipping(model.module.parameters() if use_ddp else model.parameters(), max_norm=params['gradient_clip_norm'])

    lr = learning_rate_schedule(
        curr_iter=iteration,
        max_lr=params["learning_rate"],
        min_lr=params["min_lr"],
        warm_iters=params["warmup_iters"],
        cos_iters=params["cosine_iters"]
        )
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()

    train_loss = loss.item()
    val_loss = None

    if iteration % params["eval_interval"] == 0:
        model.eval()
        val_losses = []

        with torch.no_grad():
            for _ in range(params["eval_iters"]):
                val_inputs, val_targets = data_loading(
                    val_data,
                    params["batch_size"],
                    params["context_length"],
                    params["device"]
                )
                val_logits = model(val_inputs)
                val_loss_batch = cross_entropy_loss(val_logits, val_targets)
                val_losses.append(val_loss_batch.item())
        val_loss = sum(val_losses) / len(val_losses)
        model.train()

    return train_loss, val_loss


def init_distributed():
    """Initialize distributed training environment"""
    if not dist.is_initialized():
        # These should be set by torchrun or your launch script
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    return rank, world_size


def run(params):
    # Initialize distributed training
    use_distributed = params.get("use_distributed", False)

    if use_distributed:
        rank, world_size = init_distributed()
        is_main_process = rank == 0

        # Set device for this process
        if torch.cuda.is_available():
            torch.cuda.set_device(rank)
            device = torch.device(f"cuda:{rank}")
        else:
            device = torch.device("cpu")

        params["device"] = device
    else:
        is_main_process = True
        rank = 0
        world_size = 1

    torch.manual_seed(params["seed"])
    np.random.seed(params["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(params["seed"])

    checkpoint_dir = get_checkpoint_dir(params)
    if is_main_process:
        print(f"Checkpoints will be saved to: {checkpoint_dir}")

    requested_device = params["device"]
    if not use_distributed:
        if requested_device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
        elif requested_device == "mps" and not torch.backends.mps.is_available():
            print("Warning: MPS requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device(requested_device)
        params["device"] = device

    for name, path in [("train_data", params["train_data"]), ("val_data", params["val_data"])]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} file not found: {path}")
        if os.path.getsize(path) == 0:
            raise ValueError(f"{name} file is empty: {path}")

    train_data = np.memmap(params["train_data"], dtype=params["data_dtype"], mode='r')
    val_data = np.memmap(params["val_data"], dtype=params["data_dtype"], mode='r')

    model = Transformer(
            d_model=params['d_model'],
            num_heads=params['num_heads'],
            d_ff=params['d_ff'],
            num_layers=params['num_layers'],
            vocab_size=params['vocab_size'],
            context_length=params['context_length'],
            theta=params['theta']
        )

    model = model.to(params["device"])

    # Wrap model with DDP if using distributed training
    if use_distributed:
        model = get_ddp_individual_parameters(model)

    if params["compile"]:
        model = torch.compile(model)

    # Create base optimizer
    if params["optimizer"] == "sgd":
        base_optimizer = SGDOptimizer
        optimizer_kwargs = {"lr": params["learning_rate"]}
    else:
        base_optimizer = AdamW
        optimizer_kwargs = {
            "betas": (params["beta1"], params["beta2"]),
            "weight_decay": params["weight_decay"],
            "lr": params["learning_rate"]
        }

    # Use ShardedOptimizer if distributed, otherwise regular optimizer
    if use_distributed:
        # ShardedOptimizer shards optimizer states across ranks
        optimizer = ShardedOptimizer(
            params=model.module.parameters(),
            optimizer=base_optimizer,
            **optimizer_kwargs
        )
        # Add param group to initialize the sharded optimizer
        optimizer.add_param_group({'params': model.module.parameters()})
    else:
        optimizer = base_optimizer(params=model.parameters(), **optimizer_kwargs)

    start_iter = 0
    if params["resume_from"]:
        resume_path = params["resume_from"]
        start_iter = load_checkpoint(resume_path, model.module if use_distributed else model, optimizer=optimizer, device=params["device"])

    # Only show progress bar on main process
    disable_tqdm = use_distributed and not is_main_process

    with tqdm(total=params["max_iters"], desc="Training", unit=" iters", initial=start_iter, disable=disable_tqdm) as pbar:
        for iter in range(start_iter, params["max_iters"]):
            train_loss, val_loss = pretrain(
                model=model,
                train_data=train_data,
                val_data=val_data,
                optimizer=optimizer,
                params=params,
                iteration=iter,
                use_ddp=use_distributed
                )

            # Only log from main process
            if is_main_process:
                if wandb.run is not None:
                    try:
                        log_dict = {
                            "train/loss": train_loss,
                            "lr": optimizer.param_groups[0]["lr"] if hasattr(optimizer, 'param_groups') else optimizer.optimizer.param_groups[0]["lr"],
                            "iteration": iter
                        }
                        if val_loss is not None:
                            log_dict["val/loss"] = val_loss
                        wandb.log(log_dict)
                    except Exception as e:
                        print(f"Warning: Failed to log metrics to W&B: {e}")

                if val_loss is not None:
                    pbar.set_postfix({
                        'train_loss': f'{train_loss:.4f}',
                        'val_loss': f'{val_loss:.4f}',
                        'lr': f'{optimizer.param_groups[0]["lr"] if hasattr(optimizer, "param_groups") else optimizer.optimizer.param_groups[0]["lr"]:.2e}'
                    })
                else:
                    pbar.set_postfix({
                        'train_loss': f'{train_loss:.4f}',
                        'lr': f'{optimizer.param_groups[0]["lr"] if hasattr(optimizer, "param_groups") else optimizer.optimizer.param_groups[0]["lr"]:.2e}'
                    })
                pbar.update(1)

            if iter > 0 and iter % params["checkpoint_interval"] == 0 and is_main_process:
                checkpoint_path = osp.join(checkpoint_dir, f"checkpoint_iter_{iter}.pt")
                save_checkpoint(
                    model=model.module if use_distributed else model,
                    optimizer=optimizer,
                    iteration=iter,
                    out=checkpoint_path
                )
                pbar.write(f"Saved checkpoint to {checkpoint_path}")

    if is_main_process:
        output_path = osp.join(checkpoint_dir, f"checkpoint_iter_{params['max_iters']}.pt")
        if params["max_iters"] % params["checkpoint_interval"] != 0:
            save_checkpoint(
                    model=model.module if use_distributed else model,
                    optimizer=optimizer,
                    iteration=params["max_iters"],
                    out=output_path
                )
        print(f"Model saved to {output_path}")
        wandb.finish()

    # Clean up distributed
    if use_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":

    params = get_args_pretrain()

    # Initialize wandb only on main process
    is_main = not params.get("use_distributed", False) or (dist.is_initialized() and dist.get_rank() == 0)

    if is_main and wandb.run is None:
        wandb.init(
            project="AtlasLM-Pretrain-Distributed",
            name=f"Pretrain_layers_{params['num_layers']}_bs{params['batch_size']}_lr{params['learning_rate']}",
            config=params
        )
        wandb.define_metric("iteration")
        wandb.define_metric("train/*", step_metric="iteration")
        wandb.define_metric("val/*", step_metric="iteration")


    params.update(dict(wandb.config) if wandb.run else {})

    try:
        if params.get("profile", False):
            from line_profiler import LineProfiler
            from datetime import datetime

            # Only profile on main process in distributed mode
            is_main = not params.get("use_distributed", False) or (dist.is_initialized() and dist.get_rank() == 0)

            if is_main:
                # Setup both profilers
                cprofile = cProfile.Profile()
                lp = LineProfiler()
                lp.add_function(pretrain)
                lp.add_function(run)
                lp.add_function(init_distributed)

                # Create profile output directory
                profile_output = params.get("profile_output", "profile_results/distributed_pretrain")
                profile_dir = os.path.dirname(profile_output)
                os.makedirs(profile_dir, exist_ok=True)
                base_name = os.path.splitext(os.path.basename(profile_output))[0]

                # Store profiler state globally for signal handlers
                _global_profilers['cprofile'] = cprofile
                _global_profilers['lineprofile'] = lp
                _global_profilers['profile_dir'] = profile_dir
                _global_profilers['base_name'] = base_name

                # Register signal handlers for interruption
                signal.signal(signal.SIGINT, signal_handler)
                signal.signal(signal.SIGTERM, signal_handler)
                atexit.register(save_profile_data)

                # Enable cProfile
                cprofile.enable()

                # Wrap run function with line_profiler
                lp_wrapper = lp(run)
                lp_wrapper(params)

                # Disable cProfile
                cprofile.disable()

                # Clear global state after successful completion
                _global_profilers['cprofile'] = None

                # Generate timestamp for unique filenames
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # Save cProfile results to file
                cprofile_file = os.path.join(profile_dir, f"{base_name}_cprofile_{timestamp}.txt")
                with open(cprofile_file, 'w') as f:
                    stats = pstats.Stats(cprofile, stream=f)
                    f.write("="*80 + "\n")
                    f.write("FUNCTION-LEVEL PROFILE (cProfile) - Distributed Training\n")
                    f.write("="*80 + "\n\n")
                    f.write("Top 30 functions by cumulative time:\n")
                    f.write("-"*80 + "\n")
                    stats.sort_stats('cumulative')
                    stats.print_stats(30)
                    f.write("\n\nTop 20 functions by total time:\n")
                    f.write("-"*80 + "\n")
                    stats.sort_stats('time')
                    stats.print_stats(20)

                # Save line_profiler results to file
                lineprofile_file = os.path.join(profile_dir, f"{base_name}_lineprofile_{timestamp}.txt")
                with open(lineprofile_file, 'w') as f:
                    f.write("="*80 + "\n")
                    f.write("LINE-BY-LINE PROFILE (line_profiler) - Distributed Training\n")
                    f.write("="*80 + "\n\n")
                    lp.print_stats(stream=f)

                # Also save binary cProfile data for later analysis
                cprofile_binary = os.path.join(profile_dir, f"{base_name}_cprofile_{timestamp}.prof")
                cprofile.dump_stats(cprofile_binary)

                # Print summary to console
                print("\n" + "="*80)
                print("PROFILING COMPLETE (Rank 0 / Main Process)")
                print("="*80)
                print(f"\nProfile results saved to:")
                print(f"  1. Function-level (cProfile): {cprofile_file}")
                print(f"  2. Line-by-line (line_profiler): {lineprofile_file}")
                print(f"  3. Binary cProfile data: {cprofile_binary}")
                print(f"\nTo view binary cProfile data interactively:")
                print(f"  python -m pstats {cprofile_binary}")
                print("="*80 + "\n")
            else:
                # Non-main processes just run without profiling
                run(params)
        else:
            run(params)
    except Exception as e:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        if wandb.run:
            wandb.finish(exit_code=1)
        sys.exit(1)