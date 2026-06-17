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