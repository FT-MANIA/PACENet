import os
import logging
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
import random
import shutil
import pandas as pd
from Dataset.dataset_loader import load_gait_data
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix

logger = logging.getLogger('__main__')

def setup(args):
    """loading the config and save directory"""
    config = args.__dict__
    initial_timestamp = datetime.now()
    base_dir = Path(config['output_dir'])

    plot_data_dir = base_dir / 'exp_figures'
    save_dir = base_dir / initial_timestamp.strftime("%Y-%m-%d_%H-%M")

    config['plot_data_dir'] = str(plot_data_dir)
    config['save_dir'] = str(save_dir)
    if config['run_mode'] != 'replot':
        os.makedirs(config['save_dir'], exist_ok=True)
    return config

def device_init(config):
    gpu_idx = config.get('gpu', '0')
    if torch.cuda.is_available() and gpu_idx != '-1':
        device = torch.device(f'cuda:{gpu_idx}')
        torch.cuda.set_device(device)
        logger.info(f"Using GPU: {torch.cuda.get_device_name(device)} (Index: {gpu_idx})")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    return device

def setup_seed(config):
    seed = config['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_metrics(y_true, y_pred, y_prob, loss, epoch_num, target_names=None):
    labels = list(range(len(target_names))) if target_names is not None else None

    report = classification_report(
        y_true, y_pred, labels=labels, target_names=target_names, digits=4, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    classes_idx = []
    classes_names = []
    for i, name in enumerate(target_names):
        if report[name]['support'] > 0:
            classes_idx.append(i)
            classes_names.append(name)

    n_valid = len(classes_names)
    if n_valid > 0:
        precision_macro = sum(report[name]['precision'] for name in classes_names) / n_valid
        recall_macro = sum(report[name]['recall'] for name in classes_names) / n_valid
        f1_macro = sum(report[name]['f1-score'] for name in classes_names) / n_valid
    else:
        precision_macro = recall_macro = f1_macro = 0.0

    auroc_macro = 0.0
    auprc_macro = 0.0

    n_classes = y_prob.shape[1]
    if n_classes == 2:
        auroc_macro = roc_auc_score(y_true, y_prob[:, 1])
        auprc_macro = average_precision_score(y_true, y_prob[:, 1])
    else:
        auroc_list = []
        auprc_list = []
        for c in classes_idx:
            y_true_c = (y_true == c).astype(int)
            y_prob_c = y_prob[:, c]
            if len(np.unique(y_true_c)) == 2:
                auroc_list.append(roc_auc_score(y_true_c, y_prob_c))
                auprc_list.append(average_precision_score(y_true_c, y_prob_c))

        if len(auroc_list) > 0:
            auroc_macro = sum(auroc_list) / len(auroc_list)
        if len(auprc_list) > 0:
            auprc_macro = sum(auprc_list) / len(auprc_list)

    metrics = {
        'epoch': epoch_num,
        'loss': loss,
        'accuracy': report['accuracy'],

        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,

        'auroc_macro': auroc_macro,
        'auprc_macro': auprc_macro,
        'confusion_matrix': cm
    }

    return metrics

def _parse_metric_mean(value):
    """
    Parse metric value from either a float or a string like "0.9225 ± 0.0100".
    Only the mean value before ± is used for seed qualification.
    """
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if '±' in text:
        text = text.split('±')[0].strip()
    elif '+/-' in text:
        text = text.split('+/-')[0].strip()

    try:
        return float(text)
    except ValueError:
        logger.warning(f"Cannot parse metric value: {value}")
        return None


def _get_metric_column(df, dataset_prefix, metric_name):
    """
    Find metric column in Exp_Report.csv.

    Preferred current format:
        dev_test_f1_macro_str
        ext_test_f1_macro_str

    Compatibility fallback:
        ext_test2_f1_macro_str, ext_test1_f1_macro_str
    """
    candidates = [
        f'{dataset_prefix}_{metric_name}_str'
    ]

    for col in candidates:
        if col in df.columns:
            return col

    raise KeyError(
        f"Cannot find metric column for dataset_prefix={dataset_prefix}, "
        f"metric_name={metric_name}. Available columns: {list(df.columns)}"
    )


def _judge_full_beats_all_ablations(report_df, metric_name='f1_macro'):
    """
    Judge whether Full PACENet is strictly better than all four ablations
    on both dev_test and ext_test.

    Rule:
        Full(dev_test metric) > max(other four ablations on dev_test metric)
    AND
        Full(ext_test metric) > max(other four ablations on ext_test metric)
    """
    if 'Experiment' not in report_df.columns:
        raise KeyError("Exp_Report.csv must contain an 'Experiment' column.")

    df = report_df.copy()
    df['Experiment'] = df['Experiment'].astype(str)

    full_mask = df['Experiment'].str.contains('Full PACENet', case=False, regex=False)
    if full_mask.sum() != 1:
        raise ValueError(f"Expected exactly one Full PACENet row, but found {full_mask.sum()}.")

    ablation_names = ['wo UFE', 'wo CFE', 'wo SFE', 'wo KFE']
    other_mask = pd.Series(False, index=df.index)
    for name in ablation_names:
        other_mask = other_mask | df['Experiment'].str.contains(name, case=False, regex=False)

    if other_mask.sum() == 0:
        other_mask = ~full_mask

    dev_col = _get_metric_column(df, 'dev_test', metric_name)
    ext_col = _get_metric_column(df, 'ext_test', metric_name)

    full_dev = _parse_metric_mean(df.loc[full_mask, dev_col].iloc[0])
    full_ext = _parse_metric_mean(df.loc[full_mask, ext_col].iloc[0])

    other_dev_values = [_parse_metric_mean(v) for v in df.loc[other_mask, dev_col].tolist()]
    other_ext_values = [_parse_metric_mean(v) for v in df.loc[other_mask, ext_col].tolist()]

    other_dev_values = [v for v in other_dev_values if v is not None]
    other_ext_values = [v for v in other_ext_values if v is not None]

    if full_dev is None or full_ext is None or len(other_dev_values) == 0 or len(other_ext_values) == 0:
        raise ValueError("Failed to parse valid metric values from Exp_Report.csv.")

    best_ablation_dev = max(other_dev_values)
    best_ablation_ext = max(other_ext_values)

    pass_dev = full_dev > best_ablation_dev
    pass_ext = full_ext > best_ablation_ext

    return {
        'qualified': pass_dev and pass_ext,
        'metric': metric_name,
        'dev_metric_column': dev_col,
        'ext_metric_column': ext_col,
        'full_dev': full_dev,
        'best_ablation_dev': best_ablation_dev,
        'dev_margin': full_dev - best_ablation_dev,
        'full_ext': full_ext,
        'best_ablation_ext': best_ablation_ext,
        'ext_margin': full_ext - best_ablation_ext,
        'num_ablations_compared': int(other_mask.sum())
    }


def run_seed_search(config):
    """
    Repeatedly run the existing run_exps() with seeds:
        seed_search_start, seed_search_start + 1, ..., seed_search_start + N - 1

    Keep only seeds satisfying:
        Full PACENet is strictly higher than all four ablations on dev_test
        AND
        Full PACENet is strictly higher than all four ablations on ext_test.

    Qualified seeds are saved to:
        <base_save_dir>/<seed_search_output>
    """
    start_seed = int(config.get('seed_search_start', 10))
    num_runs = int(config.get('seed_search_N', 10))
    metric_name = config.get('seed_search_metric', 'f1_macro')

    base_save_dir = config['save_dir']
    os.makedirs(base_save_dir, exist_ok=True)

    qualified_rows = []
    all_seed_rows = []

    logger.info("=" * 50)
    logger.info(f"Running seed search: start_seed={start_seed}, N={num_runs}, metric={metric_name}")
    logger.info("=" * 50)

    for i in range(num_runs):
        seed = start_seed + i

        seed_config = config.copy()
        seed_config['seed'] = seed

        # Use seed-specific folders to avoid overwriting Exp_Report.csv and checkpoints.
        seed_save_dir = os.path.join(base_save_dir, f'seed_{seed}')
        seed_config['save_dir'] = seed_save_dir
        seed_config['plot_data_dir'] = seed_save_dir
        os.makedirs(seed_save_dir, exist_ok=True)

        logger.info("=" * 50)
        logger.info(f"[Seed Search] Running seed={seed} ({i + 1}/{num_runs})")
        logger.info(f"[Seed Search] save_dir={seed_save_dir}")
        logger.info("=" * 50)

        try:
            setup_seed(seed_config)

            # Must reload data for each seed because load_gait_data() uses seed
            # to split dev/dev_test, and StratifiedKFold also uses config['seed'].
            seed_data = load_gait_data(seed_config)

            _, report_path = run_exps(seed_config, seed_data)

            report_df = pd.read_csv(report_path)
            judge = _judge_full_beats_all_ablations(report_df, metric_name=metric_name)

            row = {
                'seed': seed,
                'qualified': bool(judge['qualified']),
                'metric': judge['metric'],
                'full_dev': judge['full_dev'],
                'best_ablation_dev': judge['best_ablation_dev'],
                'dev_margin': judge['dev_margin'],
                'full_ext': judge['full_ext'],
                'best_ablation_ext': judge['best_ablation_ext'],
                'ext_margin': judge['ext_margin'],
                'num_ablations_compared': judge['num_ablations_compared'],
                'dev_metric_column': judge['dev_metric_column'],
                'ext_metric_column': judge['ext_metric_column'],
                'report_path': report_path,
                'seed_save_dir': seed_save_dir
            }

            all_seed_rows.append(row)

            if judge['qualified']:
                qualified_rows.append(row)
                logger.info(
                    f"[Seed Search] seed={seed} QUALIFIED | "
                    f"dev_margin={judge['dev_margin']:.4f}, "
                    f"ext_margin={judge['ext_margin']:.4f}"
                )
            else:
                logger.info(
                    f"[Seed Search] seed={seed} not qualified | "
                    f"dev_margin={judge['dev_margin']:.4f}, "
                    f"ext_margin={judge['ext_margin']:.4f}"
                )

                if config.get('delete_unqualified_seed_dir', False):
                    shutil.rmtree(seed_save_dir, ignore_errors=True)
                    logger.info(f"[Seed Search] Deleted unqualified seed directory: {seed_save_dir}")

        except Exception as e:
            logger.exception(f"[Seed Search] seed={seed} failed with error: {e}")
            all_seed_rows.append({
                'seed': seed,
                'qualified': False,
                'metric': metric_name,
                'error': str(e),
                'report_path': '',
                'seed_save_dir': seed_save_dir
            })

            if config.get('delete_unqualified_seed_dir', False):
                shutil.rmtree(seed_save_dir, ignore_errors=True)

    qualified_df = pd.DataFrame(qualified_rows)
    all_seed_df = pd.DataFrame(all_seed_rows)

    qualified_path = os.path.join(base_save_dir, config.get('seed_search_output', 'qualified_seeds.csv'))
    all_seed_path = os.path.join(base_save_dir, 'all_seed_search_records.csv')

    qualified_df.to_csv(qualified_path, index=False)
    all_seed_df.to_csv(all_seed_path, index=False)

    logger.info("=" * 50)
    logger.info(f"[Seed Search] Done. Qualified seeds: {len(qualified_rows)}/{num_runs}")
    logger.info(f"[Seed Search] Qualified seeds saved to: {qualified_path}")
    logger.info(f"[Seed Search] All seed records saved to: {all_seed_path}")
    logger.info("=" * 50)

    return qualified_df, all_seed_df