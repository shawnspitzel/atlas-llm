import argparse

def get_args_pretrain():
    parser = argparse.ArgumentParser('Pretrain')

    # model architecture
    parser.add_argument("--use_params", action="store_true")
    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--d_ff', type=int, default=3072)
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--vocab_size', type=int, default=50257)
    parser.add_argument('--context_length', type=int, default=1024)
    parser.add_argument('--theta', type=float, default=10000.0)

    # training config
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_iters', type=int, default=100000)
    parser.add_argument('--eval_interval', type=int, default=500)
    parser.add_argument('--eval_iters', type=int, default=100)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--checkpoint_interval', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)

    # optimizer config
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'sgd'])
    parser.add_argument('--learning_rate', '--lr', type=float, default=6e-4)
    parser.add_argument('--min_lr', type=float, default=6e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--gradient_clip_norm', type=float, default=1.0)

    # learning rate schedule
    parser.add_argument('--warmup_iters', type=int, default=2000)
    parser.add_argument('--cosine_iters', type=int, default=100000)

    # data paths
    parser.add_argument('--train_data', type=str)
    parser.add_argument('--val_data', type=str)
    parser.add_argument('--data_dtype', type=str, default='uint16')
    parser.add_argument('--checkpoint_dir', type=str)
    parser.add_argument('--resume_from', type=str, default=None)

    # wandb
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='cs336-assignment1')
    parser.add_argument('--wandb_run_name', type=str, default=None)

    # device setup
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu', 'mps'])
    parser.add_argument('--compile', type=lambda x: str(x).lower() == 'true', default=False)

    # profiling
    parser.add_argument('--profile', action='store_true', help='Enable cProfile profiling')
    parser.add_argument('--profile_output', type=str, default='checkpoints/profile_stats.prof', help='Output file for profiling stats')

    args = parser.parse_args()
    return vars(args)
