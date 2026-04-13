#!/usr/bin/env python3
"""
Comprehensive analysis and comparison of SNN grid search results.
Generates detailed CSV tables with all hyperparameters, metrics, and performance data.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import csv
import argparse
from dataclasses import dataclass
from collections import defaultdict
import statistics
import re


@dataclass
class GridResult:
    """Represents a single grid profile result."""
    filename: str
    profile_name: str
    modality: str
    loss_type: str
    learning_rate: float
    batch_size: int
    hidden_size: int
    latent_size: int
    max_batches_per_epoch: int
    configured_epochs: int
    lr_plateau_enabled: int
    final_train_loss: float
    final_val_loss: float
    mean_val_loss: float
    best_val_loss: float
    epoch_count: int
    exit_code: int
    error: str = ""
    
    @classmethod
    def from_json_file(cls, json_path: Path) -> 'GridResult':
        """Load result from JSON file."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract filename components
        filename = json_path.stem
        
        # Parse profile name to extract hyperparameters
        profile_data = data.get('profile_name', filename)
        
        # Extract metrics from JSON
        metadata = data.get('metadata', {})
        metrics = data.get('metrics', {})
        
        # Extract hyperparameters and loss type from new result format
        modality = data.get('dataset_type', 'unknown')
        loss_type = data.get('loss_type', 'unknown')
    
        # Get optimizer params
        optimizer = data.get('optimizer', {})
        lr = float(optimizer.get('learning_rate', 0.0))
    
        # Get epoch and batch info from processed data
        epoch_losses = data.get('epoch_mean_losses', [])
    
        # Extract K-fold validation losses (actual structure: kfold.fold_epoch_val_losses)
        kfold_data = data.get('kfold', {})
        fold_losses = kfold_data.get('fold_epoch_val_losses', [])
    
        # Extract final losses
        final_train_loss = epoch_losses[-1] if epoch_losses else 0.0
        final_val_loss = fold_losses[-1][-1] if fold_losses and fold_losses[-1] else 0.0
    
        # Calculate mean and best validation loss across folds
        try:
            if fold_losses and any(fold_losses):
                mean_val_loss = statistics.mean([fold[-1] for fold in fold_losses if fold])
                best_val_loss = min([fold[-1] for fold in fold_losses if fold])
            else:
                mean_val_loss = final_val_loss
                best_val_loss = final_val_loss
        except (ValueError, IndexError):
            mean_val_loss = final_val_loss
            best_val_loss = final_val_loss
    
        # Extract BS, HS, LS, MB, EP, LP from profile path / profile name
        profile_path = data.get('profile', '')
        profile_text = profile_path or profile_data or filename
        bs = 8  # Default, will extract from filename
        hs = 32  # Default, will extract from filename
        ls = 16  # Default, will extract from filename
        mb = 0
        ep = 0
        lp = 0

        # Parse tokens embedded in generated profile filenames, e.g.:
        # ..._lr0.0001_bs16_hs32_ls32_mae_mb32_ep10_lp1.json
        token_patterns = {
            'lr': r'_lr([0-9]+(?:\.[0-9]+)?)',
            'bs': r'_bs(\d+)',
            'hs': r'_hs(\d+)',
            'ls': r'_ls(\d+)',
            'mb': r'_mb(\d+)',
            'ep': r'_ep(\d+)',
            'lp': r'_lp([01])',
        }

        parsed_lr = None
        for key, pattern in token_patterns.items():
            match = re.search(pattern, profile_text)
            if not match:
                continue
            value = match.group(1)
            if key == 'lr':
                parsed_lr = float(value)
            elif key == 'bs':
                bs = int(value)
            elif key == 'hs':
                hs = int(value)
            elif key == 'ls':
                ls = int(value)
            elif key == 'mb':
                mb = int(value)
            elif key == 'ep':
                ep = int(value)
            elif key == 'lp':
                lp = int(value)

        # Prefer profile-encoded LR when available to keep config visible even for failed runs.
        if parsed_lr is not None:
            lr = parsed_lr
    
        # Fallback parsing for older filename variants.
        if 'bs' in profile_path and not re.search(r'_bs\d+', profile_text):
            try:
                for part in profile_path.split('_'):
                    if part.startswith('bs'):
                        bs = int(part[2:])
                    elif part.startswith('hs'):
                        hs = int(part[2:])
                    elif part.startswith('ls'):
                        ls = int(part[2:])
            except (ValueError, IndexError):
                pass
        
        return cls(
            filename=filename,
            profile_name=profile_data,
            modality=modality,
            loss_type=loss_type,
            learning_rate=lr,
            batch_size=bs,
            hidden_size=hs,
            latent_size=ls,
            max_batches_per_epoch=mb,
            configured_epochs=ep,
            lr_plateau_enabled=lp,
            final_train_loss=final_train_loss,
            final_val_loss=final_val_loss,
            mean_val_loss=mean_val_loss,
            best_val_loss=best_val_loss,
            epoch_count=len(epoch_losses),
            exit_code=data.get('exit_code', 1),
            error=data.get('error', '')
        )


def load_all_results(results_dir: Path) -> List[GridResult]:
    """Load all grid results from directory."""
    results = []
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return results
    
    json_files = sorted(results_dir.glob('*grid_*.json'))
    print(f"Found {len(json_files)} result files")
    
    for json_path in json_files:
        try:
            result = GridResult.from_json_file(json_path)
            results.append(result)
        except Exception as e:
            print(f"Error loading {json_path}: {e}", file=sys.stderr)
    
    return results


def generate_master_table(results: List[GridResult], output_path: Path) -> None:
    """Generate master comparison table with all profiles."""
    if not results:
        print("No results to analyze")
        return
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'Rank', 'Profile', 'Modality', 'Loss Type', 'LR', 'BS', 'HS', 'LS', 'MB', 'Configured Epochs', 'LR Plateau',
            'Final Train Loss', 'Final Val Loss', 'Mean Val Loss', 'Best Val Loss',
            'Epochs', 'Status', 'Error'
        ])
        
        # Sort by mean validation loss (best first)
        sorted_results = sorted(results, key=lambda r: r.mean_val_loss if r.exit_code == 0 else float('inf'))
        
        for rank, result in enumerate(sorted_results, 1):
            status = 'Success' if result.exit_code == 0 else f'Failed ({result.exit_code})'
            writer.writerow([
                rank,
                result.profile_name,
                result.modality,
                result.loss_type,
                f"{result.learning_rate:.4f}",
                result.batch_size,
                result.hidden_size,
                result.latent_size,
                result.max_batches_per_epoch,
                result.configured_epochs,
                result.lr_plateau_enabled,
                f"{result.final_train_loss:.6f}" if result.exit_code == 0 else 'N/A',
                f"{result.final_val_loss:.6f}" if result.exit_code == 0 else 'N/A',
                f"{result.mean_val_loss:.6f}" if result.exit_code == 0 else 'N/A',
                f"{result.best_val_loss:.6f}" if result.exit_code == 0 else 'N/A',
                result.epoch_count,
                status,
                result.error[:50] if result.error else ''
            ])


def generate_modality_comparison(results: List[GridResult], output_dir: Path) -> None:
    """Generate per-modality comparison tables."""
    by_modality = defaultdict(list)
    
    for result in results:
        if result.exit_code == 0:
            by_modality[result.modality].append(result)
    
    for modality, modality_results in sorted(by_modality.items()):
        output_path = output_dir / f'comparison_{modality}.csv'
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Rank', 'Loss Type', 'LR', 'BS', 'HS', 'LS', 'MB', 'Configured Epochs', 'LR Plateau',
                'Final Train Loss', 'Mean Val Loss', 'Best Val Loss', 'Epochs'
            ])
            
            sorted_results = sorted(modality_results, key=lambda r: r.mean_val_loss)
            
            for rank, result in enumerate(sorted_results, 1):
                writer.writerow([
                    rank,
                    result.loss_type,
                    f"{result.learning_rate:.4f}",
                    result.batch_size,
                    result.hidden_size,
                    result.latent_size,
                    result.max_batches_per_epoch,
                    result.configured_epochs,
                    result.lr_plateau_enabled,
                    f"{result.final_train_loss:.6f}",
                    f"{result.mean_val_loss:.6f}",
                    f"{result.best_val_loss:.6f}",
                    result.epoch_count
                ])


def generate_hyperparameter_sensitivity(results: List[GridResult], output_dir: Path) -> None:
    """Generate hyperparameter sensitivity analysis."""
    successful = [r for r in results if r.exit_code == 0]
    
    # Group by modality and loss type
    groups = defaultdict(list)
    for result in successful:
        key = (result.modality, result.loss_type)
        groups[key].append(result)
    
    for (modality, loss_type), group_results in sorted(groups.items()):
        output_path = output_dir / f'sensitivity_{modality}_{loss_type}.csv'
        
        # Calculate average performance per hyperparameter value
        lr_perf = defaultdict(list)
        bs_perf = defaultdict(list)
        hs_perf = defaultdict(list)
        ls_perf = defaultdict(list)
        
        for result in group_results:
            lr_perf[result.learning_rate].append(result.mean_val_loss)
            bs_perf[result.batch_size].append(result.mean_val_loss)
            hs_perf[result.hidden_size].append(result.mean_val_loss)
            ls_perf[result.latent_size].append(result.mean_val_loss)
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Hyperparameter', 'Value', 'Avg Val Loss', 'Best Val Loss', 'Count'])
            
            for param_name, param_dict in [
                ('Learning Rate', lr_perf),
                ('Batch Size', bs_perf),
                ('Hidden Size', hs_perf),
                ('Latent Size', ls_perf)
            ]:
                for value in sorted(param_dict.keys()):
                    losses = param_dict[value]
                    writer.writerow([
                        param_name,
                        value,
                        f"{statistics.mean(losses):.6f}",
                        f"{min(losses):.6f}",
                        len(losses)
                    ])


def generate_summary_stats(results: List[GridResult], output_path: Path) -> None:
    """Generate summary statistics."""
    successful = [r for r in results if r.exit_code == 0]
    failed = [r for r in results if r.exit_code != 0]
    
    stats = {
        'Total Profiles': len(results),
        'Successful': len(successful),
        'Failed': len(failed),
        'Success Rate %': 100 * len(successful) / len(results) if results else 0,
    }
    
    if successful:
        losses = [r.mean_val_loss for r in successful]
        stats.update({
            'Best Mean Val Loss': min(losses),
            'Worst Mean Val Loss': max(losses),
            'Avg Mean Val Loss': statistics.mean(losses),
            'Median Mean Val Loss': statistics.median(losses),
            'StdDev Mean Val Loss': statistics.stdev(losses) if len(losses) > 1 else 0,
        })
        
        # Per-modality stats
        by_modality = defaultdict(list)
        for result in successful:
            by_modality[result.modality].append(result.mean_val_loss)
        
        for modality in sorted(by_modality.keys()):
            losses = by_modality[modality]
            stats[f'{modality} - Best'] = min(losses)
            stats[f'{modality} - Count'] = len(losses)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for key, value in stats.items():
            if isinstance(value, float):
                writer.writerow([key, f"{value:.6f}"])
            else:
                writer.writerow([key, value])


def main():
    """Main analysis pipeline."""
    parser = argparse.ArgumentParser(
        description='Analyze SNN grid results and generate comparison CSV tables.'
    )
    parser.add_argument(
        '--results-dir',
        default='results',
        help='Directory containing result JSON files (default: results)',
    )
    parser.add_argument(
        '--output-dir',
        default='analysis',
        help='Directory where analysis CSV files are written (default: analysis)',
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("Loading grid results...")
    results = load_all_results(results_dir)
    
    if not results:
        print("No results found. Exiting.")
        return
    
    print(f"Loaded {len(results)} results. Generating analysis tables...")
    
    # Generate all analysis files
    generate_master_table(results, output_dir / 'master_comparison.csv')
    print(f"✓ Master comparison table: {output_dir / 'master_comparison.csv'}")
    
    generate_modality_comparison(results, output_dir)
    print(f"✓ Modality-specific comparisons saved to {output_dir}/")
    
    generate_hyperparameter_sensitivity(results, output_dir)
    print(f"✓ Hyperparameter sensitivity analysis saved to {output_dir}/")
    
    generate_summary_stats(results, output_dir / 'summary_statistics.csv')
    print(f"✓ Summary statistics: {output_dir / 'summary_statistics.csv'}")
    
    print("\n✓ Analysis complete!")
    print(f"  Total profiles analyzed: {len(results)}")
    print(f"  Output directory: {output_dir}/")


if __name__ == '__main__':
    main()
