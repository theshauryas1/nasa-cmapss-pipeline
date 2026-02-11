"""
Unified Training Script for Advanced Deep Learning Models

This script provides a CLI-driven interface to train TCN, BiLSTM+Attention,
and Compact Transformer models on the NASA C-MAPSS dataset.

All hyperparameters are tuned for GTX 1650 (4GB VRAM) by default.

Usage:
    # Train TCN on FD001 (recommended starting point)
    python train_advanced.py --model tcn --dataset FD001

    # Train BiLSTM+Attention on FD001
    python train_advanced.py --model bilstm --dataset FD001 --batch-size 16

    # Train Compact Transformer on FD001
    python train_advanced.py --model transformer --dataset FD001 --batch-size 8

    # Train all models sequentially
    python train_advanced.py --model all --dataset FD001

    # Custom configuration
    python train_advanced.py --model tcn --dataset FD001 --epochs 100 --seq-len 40 --no-amp

Author: Scientific Data Pipeline Project
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.ingestion import CMAPSSDataLoader, compute_training_rul, get_feature_columns
from src.data.preprocessing import DataPreprocessor
from src.models.evaluation import compute_all_metrics, print_evaluation_summary


def get_feature_cols_from_preprocessed(df):
    """Get feature columns available in the preprocessed DataFrame."""
    exclude = {'engine_id', 'cycle', 'RUL'}
    return [c for c in df.columns if c not in exclude]


def load_and_preprocess(dataset_id: str, data_dir: str = None):
    """
    Load and preprocess a C-MAPSS dataset.

    Args:
        dataset_id: One of 'FD001', 'FD002', 'FD003', 'FD004'.
        data_dir: Path to data directory.

    Returns:
        Tuple of (train_df, test_df, rul_df, feature_cols).
    """
    if data_dir is None:
        # Try common locations
        for candidate in ['data/archive/CMaps', 'data/raw', 'data']:
            if os.path.exists(candidate):
                data_dir = candidate
                break
        if data_dir is None:
            raise FileNotFoundError(
                "Could not find data directory. Use --data-dir to specify."
            )

    print(f"\n{'='*60}")
    print(f"Loading {dataset_id} from {data_dir}")
    print(f"{'='*60}")

    loader = CMAPSSDataLoader(data_dir)
    train_df, test_df, rul_df = loader.load_dataset(dataset_id)

    # Compute RUL for training data
    train_df = compute_training_rul(train_df)

    # Preprocess
    preprocessor = DataPreprocessor(
        normalization='minmax',
        rolling_windows=[5, 10],
        drop_constant_sensors=True
    )
    train_processed = preprocessor.fit_transform(train_df)

    feature_cols = get_feature_cols_from_preprocessed(train_processed)

    print(f"  Engines: {train_df['engine_id'].nunique()}")
    print(f"  Total cycles: {len(train_df)}")
    print(f"  Features: {len(feature_cols)}")

    return train_processed, test_df, rul_df, feature_cols


def train_tcn(train_df, feature_cols, args):
    """Train TCN model."""
    from src.models.tcn import TCNModel

    print(f"\n{'='*60}")
    print("[+] TRAINING: Temporal Convolutional Network (TCN)")
    print(f"{'='*60}")

    # Parse channel config
    channels = [args.channels] * args.num_blocks

    # Prepare sequences
    print(f"Preparing sequences (window={args.seq_len})...")
    X, y = TCNModel.prepare_sequences(
        train_df, feature_cols,
        sequence_length=args.seq_len
    )
    print(f"  Sequences: {X.shape[0]} | Shape: {X.shape}")

    # Create model
    model = TCNModel(
        input_dim=len(feature_cols),
        num_channels=channels,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        learning_rate=args.lr,
    )

    print(f"  Receptive field: {model.get_receptive_field()} time steps")
    param_count = sum(p.numel() for p in model.model.parameters())
    print(f"  Parameters: {param_count:,}")

    # Train
    history = model.fit(
        X, y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_amp=args.amp,
        patience=args.patience,
        gradient_clip=args.grad_clip,
    )

    return model, history, X, y


def train_bilstm(train_df, feature_cols, args):
    """Train BiLSTM+Attention model."""
    from src.models.bilstm_attention import BiLSTMAttentionModel

    print(f"\n{'='*60}")
    print("[+] TRAINING: BiLSTM + Attention")
    print(f"{'='*60}")

    # Prepare sequences
    print(f"Preparing sequences (window={args.seq_len})...")
    X, y = BiLSTMAttentionModel.prepare_sequences(
        train_df, feature_cols,
        sequence_length=args.seq_len
    )
    print(f"  Sequences: {X.shape[0]} | Shape: {X.shape}")

    # Create model
    model = BiLSTMAttentionModel(
        input_dim=len(feature_cols),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        learning_rate=args.lr,
    )

    param_count = sum(p.numel() for p in model.model.parameters())
    print(f"  Parameters: {param_count:,}")

    # Train
    batch_size = min(args.batch_size, 16)  # Cap for BiLSTM
    history = model.fit(
        X, y,
        epochs=args.epochs,
        batch_size=batch_size,
        use_amp=args.amp,
        patience=args.patience,
        gradient_clip=args.grad_clip,
    )

    return model, history, X, y


def train_transformer(train_df, feature_cols, args):
    """Train Compact Transformer model."""
    from src.models.transformer import CompactTransformerModel

    print(f"\n{'='*60}")
    print("[+] TRAINING: Compact Transformer (patience required!)")
    print(f"{'='*60}")

    # Prepare sequences
    print(f"Preparing sequences (window={args.seq_len})...")
    X, y = CompactTransformerModel.prepare_sequences(
        train_df, feature_cols,
        sequence_length=args.seq_len
    )
    print(f"  Sequences: {X.shape[0]} | Shape: {X.shape}")

    # Create model
    model = CompactTransformerModel(
        input_dim=len(feature_cols),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_ff,
        dropout=args.dropout,
        learning_rate=args.lr if args.lr != 0.001 else 0.0001,  # Lower default for Transformer
    )

    param_count = sum(p.numel() for p in model.model.parameters())
    print(f"  Parameters: {param_count:,}")

    # Train — smaller batch, mandatory AMP
    batch_size = min(args.batch_size, 8)  # Hard cap for Transformer
    history = model.fit(
        X, y,
        epochs=args.epochs,
        batch_size=batch_size,
        use_amp=True,  # Mandatory
        patience=args.patience + 5,  # Extra patience for Transformer
        gradient_clip=0.5,  # Lower clip for Transformer
    )

    return model, history, X, y


def evaluate_model(model, X, y, model_name):
    """Evaluate a trained model and print results."""
    print(f"\n--- {model_name} Evaluation ---")

    # Split off a held-out evaluation set (last 10%)
    n_eval = max(int(len(X) * 0.1), 100)
    X_eval = X[-n_eval:]
    y_eval = y[-n_eval:]

    predictions = model.predict(X_eval)
    metrics = compute_all_metrics(y_eval, predictions)

    print(f"  RMSE:       {metrics['rmse']:.2f} cycles")
    print(f"  MAE:        {metrics['mae']:.2f} cycles")
    print(f"  R²:         {metrics['r2']:.4f}")
    print(f"  NASA Score: {metrics['nasa_score']:.2f}")

    return metrics


def save_results(model, history, metrics, model_name, dataset_id, output_dir):
    """Save model checkpoint, history, and metrics."""
    os.makedirs(output_dir, exist_ok=True)

    # Save checkpoint
    checkpoint_path = os.path.join(output_dir, f"{model_name}_{dataset_id}.pt")
    model.save(checkpoint_path)
    print(f"  Checkpoint: {checkpoint_path}")

    # Save history and metrics
    results = {
        'model': model_name,
        'dataset': dataset_id,
        'timestamp': datetime.now().isoformat(),
        'metrics': {k: float(v) for k, v in metrics.items()},
        'training': {
            'epochs_trained': len(history['train_loss']),
            'best_val_loss': float(min(history['val_loss'])),
            'total_time_minutes': sum(history.get('epoch_time', [])) / 60,
        },
    }
    results_path = os.path.join(output_dir, f"{model_name}_{dataset_id}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results: {results_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train advanced deep learning models for C-MAPSS RUL prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_advanced.py --model tcn --dataset FD001
  python train_advanced.py --model bilstm --dataset FD001 --epochs 100
  python train_advanced.py --model transformer --dataset FD001 --batch-size 4
  python train_advanced.py --model all --dataset FD001
        """
    )

    # Required
    parser.add_argument('--model', type=str, required=True,
                        choices=['tcn', 'bilstm', 'transformer', 'all'],
                        help='Model to train')
    parser.add_argument('--dataset', type=str, default='FD001',
                        choices=['FD001', 'FD002', 'FD003', 'FD004'],
                        help='C-MAPSS sub-dataset (default: FD001)')

    # Data
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to data directory')
    parser.add_argument('--seq-len', type=int, default=30,
                        help='Sequence/window length (default: 30)')

    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate (default: 0.2)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience (default: 10)')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='Gradient clipping norm (default: 1.0)')
    parser.add_argument('--no-amp', dest='amp', action='store_false',
                        help='Disable mixed precision (AMP)')
    parser.set_defaults(amp=True)

    # TCN specific
    parser.add_argument('--channels', type=int, default=64,
                        help='TCN channel width (default: 64)')
    parser.add_argument('--num-blocks', type=int, default=4,
                        help='TCN blocks / BiLSTM layers (default: 4)')
    parser.add_argument('--kernel-size', type=int, default=3,
                        help='TCN kernel size (default: 3)')

    # BiLSTM specific
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='BiLSTM hidden dimension (default: 128)')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='BiLSTM/Transformer layers (default: 2)')

    # Transformer specific
    parser.add_argument('--d-model', type=int, default=128,
                        help='Transformer d_model (default: 128, max 256)')
    parser.add_argument('--nhead', type=int, default=4,
                        help='Transformer attention heads (default: 4)')
    parser.add_argument('--dim-ff', type=int, default=256,
                        help='Transformer FFN dim (default: 256)')

    # Output
    parser.add_argument('--output-dir', type=str, default='checkpoints',
                        help='Output directory for checkpoints (default: checkpoints)')

    args = parser.parse_args()

    # Banner
    print("\n" + "=" * 60)
    print("  NASA C-MAPSS Advanced Model Training")
    print("  GTX 1650 Optimized Configuration")
    print("=" * 60)
    print(f"  Model:    {args.model}")
    print(f"  Dataset:  {args.dataset}")
    print(f"  Epochs:   {args.epochs}")
    print(f"  AMP:      {args.amp}")
    print(f"  Seq len:  {args.seq_len}")

    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  GPU:      {gpu} ({mem:.1f} GB)")
        else:
            print("  GPU:      None (CPU mode)")
    except ImportError:
        print("  GPU:      PyTorch not installed")
        sys.exit(1)

    # Load data
    train_df, test_df, rul_df, feature_cols = load_and_preprocess(
        args.dataset, args.data_dir
    )

    # Train model(s)
    models_to_train = (
        ['tcn', 'bilstm', 'transformer'] if args.model == 'all'
        else [args.model]
    )

    all_results = {}

    for model_name in models_to_train:
        start_time = time.time()

        try:
            if model_name == 'tcn':
                model, history, X, y = train_tcn(train_df, feature_cols, args)
            elif model_name == 'bilstm':
                model, history, X, y = train_bilstm(train_df, feature_cols, args)
            elif model_name == 'transformer':
                model, history, X, y = train_transformer(train_df, feature_cols, args)

            # Evaluate
            metrics = evaluate_model(model, X, y, model_name.upper())

            # Save
            save_results(model, history, metrics, model_name, args.dataset, args.output_dir)

            elapsed = time.time() - start_time
            print(f"\n  Total time: {elapsed/60:.1f} minutes")

            all_results[model_name] = metrics

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n[X] {model_name.upper()} OOM! Try reducing:")
                print(f"   --batch-size, --seq-len, --channels, --d-model")
                if hasattr(torch.cuda, 'empty_cache'):
                    import torch
                    torch.cuda.empty_cache()
            else:
                raise

    # Summary
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("  MODEL COMPARISON")
        print(f"{'='*60}")
        print(f"{'Model':<15} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'NASA':>10}")
        print("-" * 51)
        for name, metrics in all_results.items():
            print(
                f"{name.upper():<15} "
                f"{metrics['rmse']:>8.2f} "
                f"{metrics['mae']:>8.2f} "
                f"{metrics['r2']:>8.4f} "
                f"{metrics['nasa_score']:>10.2f}"
            )


if __name__ == '__main__':
    main()
