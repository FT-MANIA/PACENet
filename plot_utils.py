import os
import logging
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import pandas as pd
from collections import defaultdict
from utils import compute_metrics
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, f1_score, average_precision_score, precision_recall_curve
from sklearn.preprocessing import label_binarize
from scipy.ndimage import gaussian_filter1d
from scipy.stats import wilcoxon

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def stat_kfold_metrics(config, dev_test_reports, ext_test_reports):
    fold_metrics_data = []
    k = config['k_folds']
    for f in range(k):
        fold_metrics_data.append({
            'Fold': f + 1,
            'Dev_test_Accuracy': dev_test_reports[f]['accuracy'],
            'Dev_test_Precision_Macro': dev_test_reports[f].get('precision_macro', dev_test_reports[f]['precision_macro']),
            'Dev_test_Recall_Macro': dev_test_reports[f].get('recall_macro', dev_test_reports[f]['recall_macro']),
            'Dev_test_F1_Macro': dev_test_reports[f].get('f1_macro', dev_test_reports[f]['f1_macro']),
            'Dev_test_AUROC_Macro': dev_test_reports[f].get('auroc_macro', 0.0),
            'Dev_test_AUPRC_Macro': dev_test_reports[f].get('auprc_macro', 0.0),

            'Ext_test_Accuracy': ext_test_reports[f]['accuracy'],
            'Ext_test_Precision_Macro': ext_test_reports[f].get('precision_macro', ext_test_reports[f]['precision_macro']),
            'Ext_test_Recall_Macro': ext_test_reports[f].get('recall_macro', ext_test_reports[f]['recall_macro']),
            'Ext_test_F1_Macro': ext_test_reports[f].get('f1_macro', ext_test_reports[f]['f1_macro']),
            'Ext_test_AUROC_Macro': ext_test_reports[f].get('auroc_macro', 0.0),
            'Ext_test_AUPRC_Macro': ext_test_reports[f].get('auprc_macro', 0.0)
        })
    df_metrics = pd.DataFrame(fold_metrics_data)

    mean_row = {k: df_metrics[k].mean() if k != 'Fold' else 'Mean' for k in df_metrics.columns}
    std_row = {k: df_metrics[k].std(ddof=0) if k != 'Fold' else 'Std' for k in df_metrics.columns}
    df_metrics = pd.concat([df_metrics, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    csv_path = os.path.join(config['save_dir'], 'fold_metrics_summary.csv')
    df_metrics.to_csv(csv_path, index=False)

def subset_extractor(ext_test_reports, dev_test_reports, class_names):
    ext_test_subset_reports, dev_test_subset_reports = defaultdict(list), defaultdict(list)
    for reports, subset_dict in zip([ext_test_reports, dev_test_reports], [ext_test_subset_reports, dev_test_subset_reports]):
        for rep in reports:
            y_true, y_pred, y_prob, y_demo = rep['y_true'], rep['y_pred'], rep['y_prob'], rep['y_demo']
            masks = {
                'All': np.ones(len(y_demo), dtype=bool),
                'Male': y_demo[:, 0] == 1, 'Female': y_demo[:, 0] == 0,
                'Age_under_40': y_demo[:, 1] < 40, 'Age_over_40': y_demo[:, 1] >= 40,
                'Age_under_50': y_demo[:, 1] < 50, 'Age_over_50': y_demo[:, 1] >= 50,
                'Age_under_60': y_demo[:, 1] < 60, 'Age_over_60': y_demo[:, 1] >= 60,
            }
            for sub_name, mask in masks.items():
                if np.sum(mask) == 0: continue
                sub_metrics = compute_metrics(y_true[mask], y_pred[mask], y_prob[mask], loss=0.0,
                                              epoch_num=0, target_names=class_names)
                subset_dict[sub_name].append(sub_metrics)
    return ext_test_subset_reports, dev_test_subset_reports

def subset_analysis(subset_reports_dict, save_dir, prefix='Test'):
    for sub_name, reports in subset_reports_dict.items():
        if len(reports) == 0: continue
        sub_metrics_data = []
        for f_idx, rep in enumerate(reports):
            sub_metrics_data.append({
                'Fold': f_idx + 1,
                'Accuracy': rep['accuracy'],
                'Precision_Macro': rep.get('precision_macro', 0.0),
                'Recall_Macro': rep.get('recall_macro', 0.0),
                'F1_Macro': rep.get('f1_macro', 0.0),
                'AUROC_Macro': rep.get('auroc_macro', 0.0),
                'AUPRC_Macro': rep.get('auprc_macro', 0.0)
            })
        df_sub = pd.DataFrame(sub_metrics_data)

        mean_row = {k: df_sub[k].mean() if k != 'Fold' else 'Mean' for k in df_sub.columns}
        std_row = {k: df_sub[k].std(ddof=0) if k != 'Fold' else 'Std' for k in df_sub.columns}
        df_sub = pd.concat([df_sub, pd.DataFrame([mean_row, std_row])], ignore_index=True)

        csv_path = os.path.join(save_dir, f'{prefix}_Metrics_Subset_{sub_name}.csv')
        df_sub.to_csv(csv_path, index=False)
        logger.info(f"The detailed indicators of the subgroup [{sub_name}] under {prefix} have been saved to: {csv_path}")

def attn_weight_extractor(results, device, class_names):
    attn_weight = []
    model, loader = results['best_model'], results['ext_test_loader']
    model.eval()
    found_counts = {0: 0, 1: 0, 2: 0}
    target = {
        'Healthy': ['23'],
        'ACLD': ['69'],
        'KOA': ['18']
    }

    with torch.no_grad():
        for batch in loader:
            x, targets, demo, raw_kf, trace_info, indices = batch
            x, demo, raw_kf, targets = x.to(device), demo.to(device), raw_kf.to(device), targets.to(device)
            captured_attns = []

            def attention_hook(module, input, output):
                captured_attns.append(module.attn_weights.clone().cpu().numpy())

            handle = model.model.UFE.attention_layer.register_forward_hook(attention_hook)
            logits = model(x, raw_kf)
            handle.remove()
            preds = torch.argmax(logits, dim=1)

            for i in range(len(targets)):
                true_label, pred_label = targets[i].item(), preds[i].item()
                label_name = class_names[true_label]

                orig_id = trace_info.get('original_id', [])[i]
                acld_side = trace_info.get('acld_side', [])[i]

                is_target = False
                if orig_id in target[label_name] and true_label == pred_label: is_target = True

                if is_target:
                    attn_left_2d, attn_right_2d = captured_attns[0][i], captured_attns[1][i]
                    attn_left_1d = np.mean(
                        np.mean(attn_left_2d, axis=0) if len(attn_left_2d.shape) == 3 else attn_left_2d, axis=0)
                    attn_right_1d = np.mean(
                        np.mean(attn_right_2d, axis=0) if len(attn_right_2d.shape) == 3 else attn_right_2d,
                        axis=0)

                    image_filename = f'Bilateral_Attention_Class_{label_name}_ID_{orig_id}.png'

                    attn_weight.append({
                        'raw_signal': x[i].cpu().numpy(),
                        'attn_left_1d': attn_left_1d,
                        'attn_right_1d': attn_right_1d,
                        'image_filename': image_filename,
                        'label_name': label_name,
                        'acld_side': acld_side,
                        'orig_id': orig_id
                    })
                    found_counts[true_label] += 1
                    target[label_name].remove(orig_id)
    return attn_weight

def plot_kfold_confusion_matrix(cms, class_names, save_dir, title, name='kfold_overall_cm.png'):
    cms = np.array(cms)
    K, n_classes, _ = cms.shape
    cm_sum = np.sum(cms, axis=0)

    cms_norm = np.zeros_like(cms, dtype=float)
    for i in range(K):
        row_sums = cms[i].sum(axis=1)[:, np.newaxis]
        row_sums[row_sums == 0] = 1
        cms_norm[i] = cms[i] / row_sums

    cm_mean_pct = np.mean(cms_norm, axis=0) * 100
    cm_std_pct = np.std(cms_norm, axis=0) * 100

    annot_data = np.empty((n_classes, n_classes), dtype=object)
    for i in range(n_classes):
        for j in range(n_classes):
            annot_data[i, j] = f"{cm_mean_pct[i, j]:.1f} ± {cm_std_pct[i, j]:.1f}%\n({int(cm_sum[i, j])})"

    plt.figure(figsize=(8, 6))

    sns.heatmap(cm_mean_pct, annot=annot_data, fmt='', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                vmin=0, vmax=100, annot_kws={"size": 11, "weight": "bold"})

    plt.title(title, fontsize=14, pad=15, weight='bold')
    plt.ylabel('True Label', fontsize=12, weight='bold')
    plt.xlabel('Predicted Label', fontsize=12, weight='bold')
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11, rotation=0)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, name), dpi=600, bbox_inches='tight')
    plt.close()

    logger.info(f"K-Fold confusion matrix saved to: {os.path.join(save_dir, name)}")

def plot_class_spectral_energy(x, y, class_names, fs, save_dir, name='class_spectral_energy_comparison.png'):
    logger = logging.getLogger(__name__)

    import matplotlib as mpl
    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['font.serif'] = ['Times New Roman']
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['axes.unicode_minus'] = False

    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)

    B, C, T = x.shape

    window = torch.hamming_window(T).to(x.device)
    x_windowed = x * window.view(1, 1, -1)
    x_fft = torch.fft.rfft(x_windowed, dim=-1)

    x_mag = torch.abs(x_fft)
    x_log = torch.log(x_mag + 1e-6)  # [B, Channels, Freqs]

    x_log_global = torch.mean(x_log, dim=1).cpu().numpy()

    freqs = np.fft.rfftfreq(T, d=1 / fs)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', 's', '^']

    y_min = np.min(x_log_global) - 0.5

    line_handles, line_labels = [], []
    star_handles, star_labels = [], []

    for i, class_name in enumerate(class_names):
        idx = (y == i)
        if np.sum(idx) == 0:
            continue

        class_mean_energy = np.mean(x_log_global[idx], axis=0)

        line, = ax.plot(freqs, class_mean_energy, marker=markers[i % len(markers)],
                        color=colors[i % len(colors)], linewidth=2.5, markersize=5,
                        label=class_name)
        line_handles.append(line)
        line_labels.append(class_name)

        ax.fill_between(freqs, y_min, class_mean_energy,
                        color=colors[i % len(colors)], alpha=0.15)

        f_min, f_max = 0.1, 5.0

        search_mask = (freqs > f_min) & (freqs < f_max)
        search_indices = np.where(search_mask)[0]

        if len(search_indices) > 0:
            local_energies = class_mean_energy[search_indices]
            local_peak_idx = search_indices[np.argmax(local_energies)]

            peak_f = freqs[local_peak_idx]
            peak_e = class_mean_energy[local_peak_idx]

            star, = ax.plot(peak_f, peak_e, marker='*', color=colors[i % len(colors)],
                            markersize=11, markeredgecolor='black', markeredgewidth=1.0,
                            linestyle='None', zorder=10)

            star_handles.append(star)
            star_labels.append(f'{class_name}: f ≈ {peak_f:.1f} Hz, E ≈ {peak_e:.1f}')

    ax.set_xlabel('Frequency (Hz)', fontsize=14, weight='bold')
    ax.set_ylabel('Mean Log-Spectral Energy', fontsize=14, weight='bold')

    ax.set_ylim(bottom=y_min)
    ax.set_xlim(left=0, right=min(30, np.max(freqs)))
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.setp(ax.get_xticklabels(), fontsize=12, fontweight='bold')
    plt.setp(ax.get_yticklabels(), fontsize=12, fontweight='bold')

    leg1 = ax.legend(handles=line_handles, labels=line_labels,
                     title='Clinical Group', title_fontsize='13', fontsize='12',
                     loc='upper right', framealpha=0.95, edgecolor='black')
    ax.add_artist(leg1)

    if star_handles:
        leg2 = ax.legend(handles=star_handles, labels=star_labels,
                         title='Dominant Spectral Peaks', title_fontsize='12', fontsize='11',
                         loc='upper right', bbox_to_anchor=(1.0, 0.72),
                         framealpha=0.95, edgecolor='black')

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, name)
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    logger.info(f"Spectral energy distribution figures have been saved to: {save_path}")

def plot_multi_model_roc_comparison(benchmark_results_dict, n_classes, save_dir, model='test', name='multi_model_roc_comparison.png'):
    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['font.serif'] = ['Times New Roman']
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(10, 8))
    mean_fpr = np.linspace(0, 1, 100)

    base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b',
                   '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8']

    color_idx = 0
    legend_data = []

    for model_name, data in benchmark_results_dict.items():
        if model_name == 'SVM' or model_name == 'XGBoost':
            continue
        y_true_list = data[f'{model}_y_true_list']
        y_prob_list = data[f'{model}_y_prob_list']
        k_folds = len(y_true_list)

        fold_aucs = []
        all_classes_mean_tpr = []

        for i in range(k_folds):
            y_true = y_true_list[i]
            y_prob = y_prob_list[i]

            if n_classes == 2:
                y_true_bin = (y_true == 1).astype(int)
                y_prob_pos = y_prob[:, 1]
                if len(np.unique(y_true_bin)) >= 2:
                    fold_aucs.append(roc_auc_score(y_true_bin, y_prob_pos))
            else:
                valid_aucs = []
                for c in range(n_classes):
                    y_true_c = (y_true == c).astype(int)
                    y_prob_c = y_prob[:, c]
                    if len(np.unique(y_true_c)) >= 2:
                        valid_aucs.append(roc_auc_score(y_true_c, y_prob_c))
                if valid_aucs:
                    fold_aucs.append(np.mean(valid_aucs))

        real_auc_mean = np.mean(fold_aucs)

        classes_to_plot = [1] if n_classes == 2 else range(n_classes)

        for c in classes_to_plot:
            tprs = []
            for i in range(k_folds):
                y_true = y_true_list[i]
                y_prob = y_prob_list[i]
                y_true_bin = (y_true == c).astype(int)
                y_prob_c = y_prob[:, c]

                if len(np.unique(y_true_bin)) < 2: continue

                fpr, tpr, _ = roc_curve(y_true_bin, y_prob_c)
                interp_tpr = np.interp(mean_fpr, fpr, tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)

            if len(tprs) > 0:
                mean_tpr = np.mean(tprs, axis=0)
                mean_tpr[-1] = 1.0
                all_classes_mean_tpr.append(mean_tpr)

        if len(all_classes_mean_tpr) > 0:
            final_mean_tpr = np.mean(all_classes_mean_tpr, axis=0)
            final_mean_tpr[-1] = 1.0

            is_ours = 'Ours' in model_name or 'PACENet' in model_name
            color = '#d62728' if is_ours else base_colors[color_idx % len(base_colors)]
            lw = 4.5 if is_ours else 3.0
            zorder = 10 if is_ours else 5

            if not is_ours: color_idx += 1

            if is_ours: model_name = 'PACENet'
            legend_data.append({
                'name': model_name,
                'auc': real_auc_mean,
                'fpr': mean_fpr,
                'tpr': final_mean_tpr,
                'color': color,
                'lw': lw,
                'zorder': zorder,
                'is_ours': is_ours
            })

    legend_data.sort(key=lambda x: x['auc'], reverse=True)

    for item in legend_data:
        plt.plot(item['fpr'], item['tpr'], label=f"{item['name']} ({item['auc']:.4f})",
                 color=item['color'], lw=item['lw'], zorder=item['zorder'],
                 alpha=0.9 if item['is_ours'] else 0.8)

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', label='Chance', alpha=.8)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])

    plt.xlabel('False Positive Rate (1 − Specificity)', fontsize=16, weight='bold')
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=16, weight='bold')

    plt.xticks(np.arange(0, 1.1, 0.2), fontsize=12)
    plt.yticks(np.arange(0, 1.1, 0.2), fontsize=12)

    legend = plt.legend(loc="lower right", fontsize=18, framealpha=0.95, edgecolor='black', fancybox=True)
    for text, item in zip(legend.get_texts(), legend_data):
        if 'PACENet' in item['name']:
            text.set_fontweight('bold')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, name)
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()

def plot_both_dev_ext_roc_comparison(benchmark_results_dict, n_classes, save_dir, name):
    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['font.serif'] = ['Times New Roman']
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(18, 11))
    mean_fpr = np.linspace(0, 1, 100)

    target_models = ['PACENet', 'SVM', 'XGBoost']

    model_colors = {
        'PACENet': '#d62728',
        'SVM': '#1f77b4',
        'XGBoost': '#ff7f0e'
    }

    dataset_styles = [
        {
            'key': 'dev_test',
            'label': 'Internal',
            'linestyle': '-',
            'alpha': 0.95
        },
        {
            'key': 'ext_test',
            'label': 'External',
            'linestyle': '--',
            'alpha': 0.65
        }
    ]

    legend_data = []

    for dataset_info in dataset_styles:
        dataset_key = dataset_info['key']
        dataset_label = dataset_info['label']
        linestyle = dataset_info['linestyle']
        alpha = dataset_info['alpha']

        for model_name, data in benchmark_results_dict.items():

            if model_name not in target_models:
                continue

            y_true_key = f'{dataset_key}_y_true_list'
            y_prob_key = f'{dataset_key}_y_prob_list'

            if y_true_key not in data or y_prob_key not in data:
                print(f"[Warning] {model_name} does not contain {y_true_key} or {y_prob_key}, skip.")
                continue

            y_true_list = data[y_true_key]
            y_prob_list = data[y_prob_key]
            k_folds = len(y_true_list)

            fold_aucs = []
            all_classes_mean_tpr = []

            for i in range(k_folds):
                y_true = y_true_list[i]
                y_prob = y_prob_list[i]

                if n_classes == 2:
                    y_true_bin = (y_true == 1).astype(int)
                    y_prob_pos = y_prob[:, 1]

                    if len(np.unique(y_true_bin)) >= 2:
                        fold_aucs.append(roc_auc_score(y_true_bin, y_prob_pos))

                else:
                    valid_aucs = []
                    for c in range(n_classes):
                        y_true_c = (y_true == c).astype(int)
                        y_prob_c = y_prob[:, c]

                        if len(np.unique(y_true_c)) >= 2:
                            valid_aucs.append(roc_auc_score(y_true_c, y_prob_c))

                    if valid_aucs:
                        fold_aucs.append(np.mean(valid_aucs))

            real_auc_mean = np.mean(fold_aucs) if len(fold_aucs) > 0 else np.nan

            classes_to_plot = [1] if n_classes == 2 else range(n_classes)

            for c in classes_to_plot:
                tprs = []

                for i in range(k_folds):
                    y_true = y_true_list[i]
                    y_prob = y_prob_list[i]

                    y_true_bin = (y_true == c).astype(int)
                    y_prob_c = y_prob[:, c]

                    if len(np.unique(y_true_bin)) < 2:
                        continue

                    fpr, tpr, _ = roc_curve(y_true_bin, y_prob_c)
                    interp_tpr = np.interp(mean_fpr, fpr, tpr)
                    interp_tpr[0] = 0.0
                    tprs.append(interp_tpr)

                if len(tprs) > 0:
                    mean_tpr = np.mean(tprs, axis=0)
                    mean_tpr[-1] = 1.0
                    all_classes_mean_tpr.append(mean_tpr)

            if len(all_classes_mean_tpr) > 0:
                final_mean_tpr = np.mean(all_classes_mean_tpr, axis=0)
                final_mean_tpr[-1] = 1.0

                is_ours = 'Ours' in model_name or 'PACENet' in model_name

                legend_data.append({
                    'name': 'PACENet' if is_ours else model_name,
                    'dataset_label': dataset_label,
                    'auc': real_auc_mean,
                    'fpr': mean_fpr,
                    'tpr': final_mean_tpr,
                    'color': model_colors.get(model_name, '#7f7f7f'),
                    'linestyle': linestyle,
                    'alpha': alpha,
                    'lw': 9.0 if is_ours else 6.0,
                    'zorder': 10 if is_ours else 5,
                    'is_ours': is_ours
                })

    dataset_order = {
        'Internal': 0,
        'External': 1
    }

    model_order = {
        'PACENet': 0,
        'SVM': 1,
        'XGBoost': 2
    }

    legend_data.sort(
        key=lambda x: (
            dataset_order.get(x['dataset_label'], 99),
            model_order.get(x['name'], 99)
        )
    )

    for item in legend_data:
        plt.plot(
            item['fpr'],
            item['tpr'],
            label=f"{item['name']} {item['dataset_label']} ({item['auc']:.4f})",
            color=item['color'],
            linestyle=item['linestyle'],
            lw=item['lw'],
            zorder=item['zorder'],
            alpha=item['alpha']
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle='--',
        lw=2,
        color='gray',
        label='Chance',
        alpha=.8
    )

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])

    plt.xlabel('False Positive Rate (1 − Specificity)', fontsize=24, weight='bold')
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=24, weight='bold')

    plt.xticks(np.arange(0, 1.1, 0.2), fontsize=20)
    plt.yticks(np.arange(0, 1.1, 0.2), fontsize=20)

    legend = plt.legend(
        loc="lower right",
        fontsize=28,
        framealpha=0.95,
        edgecolor='black',
        fancybox=True
    )

    for text in legend.get_texts():
        if 'PACENet' in text.get_text():
            text.set_fontweight('bold')

    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, name)
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()

def plot_multi_model_pr_comparison(benchmark_results_dict, n_classes, save_dir, model='test', name='multi_model_pr_comparison.png'):
    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['font.serif'] = ['Times New Roman']
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(10, 8))
    mean_recall = np.linspace(0, 1, 100)

    base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b',
                   '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8']

    color_idx = 0
    legend_data = []

    for model_name, data in benchmark_results_dict.items():
        if model_name == 'SVM' or model_name == 'XGBoost':
            continue
        y_true_list = data[f'{model}_y_true_list']
        y_prob_list = data[f'{model}_y_prob_list']
        k_folds = len(y_true_list)

        fold_aps = []
        all_classes_mean_precision = []

        for i in range(k_folds):
            y_true = y_true_list[i]
            y_prob = y_prob_list[i]

            if n_classes == 2:
                y_true_bin = (y_true == 1).astype(int)
                y_prob_pos = y_prob[:, 1]
                if len(np.unique(y_true_bin)) >= 2:
                    fold_aps.append(average_precision_score(y_true_bin, y_prob_pos))
            else:
                valid_aps = []
                for c in range(n_classes):
                    y_true_c = (y_true == c).astype(int)
                    y_prob_c = y_prob[:, c]
                    if len(np.unique(y_true_c)) >= 2:
                        valid_aps.append(average_precision_score(y_true_c, y_prob_c))
                if valid_aps:
                    fold_aps.append(np.mean(valid_aps))

        real_ap_mean = np.mean(fold_aps)

        classes_to_plot = [1] if n_classes == 2 else range(n_classes)

        for c in classes_to_plot:
            prs = []
            for i in range(k_folds):
                y_true = y_true_list[i]
                y_prob = y_prob_list[i]
                y_true_bin = (y_true == c).astype(int)
                y_prob_c = y_prob[:, c]

                if len(np.unique(y_true_bin)) < 2: continue

                precision, recall, _ = precision_recall_curve(y_true_bin, y_prob_c)
                interp_pr = np.interp(mean_recall, recall[::-1], precision[::-1])
                prs.append(interp_pr)

            if len(prs) > 0:
                mean_precision = np.mean(prs, axis=0)
                all_classes_mean_precision.append(mean_precision)

        if len(all_classes_mean_precision) > 0:
            final_mean_precision = np.mean(all_classes_mean_precision, axis=0)

            is_ours = 'Ours' in model_name or 'PACENet' in model_name
            color = '#d62728' if is_ours else base_colors[color_idx % len(base_colors)]
            lw = 4.5 if is_ours else 3.0
            zorder = 10 if is_ours else 5

            if not is_ours: color_idx += 1

            if is_ours: model_name = 'PACENet'
            legend_data.append({
                'name': model_name,
                'ap': real_ap_mean,
                'recall': mean_recall,
                'precision': final_mean_precision,
                'color': color,
                'lw': lw,
                'zorder': zorder,
                'is_ours': is_ours
            })

    legend_data.sort(key=lambda x: x['ap'], reverse=True)

    for item in legend_data:
        plt.plot(item['recall'], item['precision'], label=f"{item['name']} ({item['ap']:.4f})",
                 color=item['color'], lw=item['lw'], zorder=item['zorder'],
                 alpha=0.9 if item['is_ours'] else 0.8)

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])

    plt.xlabel('Recall', fontsize=16, weight='bold')
    plt.ylabel('Precision', fontsize=16, weight='bold')

    plt.xticks(np.arange(0, 1.1, 0.2), fontsize=12)
    plt.yticks(np.arange(0, 1.1, 0.2), fontsize=12)

    legend = plt.legend(loc="lower left", fontsize=18, framealpha=0.95, edgecolor='black', fancybox=True)
    for text, item in zip(legend.get_texts(), legend_data):
        if 'PACENet' in item['name']:
            text.set_fontweight('bold')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, name)
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()

def plot_both_dev_ext_pr_comparison(benchmark_results_dict, n_classes, save_dir, name):
    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['font.serif'] = ['Times New Roman']
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(18, 11))
    mean_recall = np.linspace(0, 1, 100)

    target_models = ['PACENet', 'SVM', 'XGBoost']

    model_colors = {
        'PACENet': '#d62728',
        'SVM': '#1f77b4',
        'XGBoost': '#ff7f0e'
    }

    dataset_styles = [
        {
            'key': 'dev_test',
            'label': 'Internal',
            'linestyle': '-',
            'alpha': 0.95
        },
        {
            'key': 'ext_test',
            'label': 'External',
            'linestyle': '--',
            'alpha': 0.65
        }
    ]

    legend_data = []

    for dataset_info in dataset_styles:
        dataset_key = dataset_info['key']
        dataset_label = dataset_info['label']
        linestyle = dataset_info['linestyle']
        alpha = dataset_info['alpha']

        for model_name, data in benchmark_results_dict.items():

            if model_name not in target_models:
                continue

            y_true_key = f'{dataset_key}_y_true_list'
            y_prob_key = f'{dataset_key}_y_prob_list'

            if y_true_key not in data or y_prob_key not in data:
                print(f"[Warning] {model_name} does not contain {y_true_key} or {y_prob_key}, skip.")
                continue

            y_true_list = data[y_true_key]
            y_prob_list = data[y_prob_key]
            k_folds = len(y_true_list)

            fold_aps = []
            all_classes_mean_precision = []

            for i in range(k_folds):
                y_true = y_true_list[i]
                y_prob = y_prob_list[i]

                if n_classes == 2:
                    y_true_bin = (y_true == 1).astype(int)
                    y_prob_pos = y_prob[:, 1]

                    if len(np.unique(y_true_bin)) >= 2:
                        fold_aps.append(
                            average_precision_score(y_true_bin, y_prob_pos)
                        )

                else:
                    valid_aps = []

                    for c in range(n_classes):
                        y_true_c = (y_true == c).astype(int)
                        y_prob_c = y_prob[:, c]

                        if len(np.unique(y_true_c)) >= 2:
                            valid_aps.append(
                                average_precision_score(y_true_c, y_prob_c)
                            )

                    if valid_aps:
                        fold_aps.append(np.mean(valid_aps))

            real_ap_mean = np.mean(fold_aps) if len(fold_aps) > 0 else np.nan

            classes_to_plot = [1] if n_classes == 2 else range(n_classes)

            for c in classes_to_plot:
                prs = []

                for i in range(k_folds):
                    y_true = y_true_list[i]
                    y_prob = y_prob_list[i]

                    y_true_bin = (y_true == c).astype(int)
                    y_prob_c = y_prob[:, c]

                    if len(np.unique(y_true_bin)) < 2:
                        continue

                    precision, recall, _ = precision_recall_curve(
                        y_true_bin,
                        y_prob_c
                    )

                    interp_precision = np.interp(
                        mean_recall,
                        recall[::-1],
                        precision[::-1]
                    )

                    prs.append(interp_precision)

                if len(prs) > 0:
                    mean_precision = np.mean(prs, axis=0)
                    all_classes_mean_precision.append(mean_precision)

            if len(all_classes_mean_precision) > 0:
                final_mean_precision = np.mean(all_classes_mean_precision, axis=0)

                is_ours = 'Ours' in model_name or 'PACENet' in model_name

                legend_data.append({
                    'name': 'PACENet' if is_ours else model_name,
                    'dataset_label': dataset_label,
                    'ap': real_ap_mean,
                    'recall': mean_recall,
                    'precision': final_mean_precision,
                    'color': model_colors.get(model_name, '#7f7f7f'),
                    'linestyle': linestyle,
                    'alpha': alpha,
                    'lw': 7.0 if is_ours else 4.0,
                    'zorder': 10 if is_ours else 5,
                    'is_ours': is_ours
                })

    dataset_order = {
        'Internal': 0,
        'External': 1
    }

    model_order = {
        'PACENet': 0,
        'SVM': 1,
        'XGBoost': 2
    }

    legend_data.sort(
        key=lambda x: (
            dataset_order.get(x['dataset_label'], 99),
            model_order.get(x['name'], 99)
        )
    )

    for item in legend_data:
        plt.plot(
            item['recall'],
            item['precision'],
            label=f"{item['name']} {item['dataset_label']} ({item['ap']:.4f})",
            color=item['color'],
            linestyle=item['linestyle'],
            lw=item['lw'],
            zorder=item['zorder'],
            alpha=item['alpha']
        )

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])

    plt.xlabel('Recall', fontsize=24, weight='bold')
    plt.ylabel('Precision', fontsize=24, weight='bold')

    plt.xticks(np.arange(0, 1.1, 0.2), fontsize=20)
    plt.yticks(np.arange(0, 1.1, 0.2), fontsize=20)

    legend = plt.legend(
        loc="lower left",
        fontsize=28,
        framealpha=0.95,
        edgecolor='black',
        fancybox=True
    )

    for text in legend.get_texts():
        if 'PACENet' in text.get_text():
            text.set_fontweight('bold')

    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, name)
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()

def plot_multi_model_metric_boxplots(benchmark_results_dict, n_classes, save_dir, model='test', metric='f1', name=None, threshold=0.5, average='macro'):
    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['font.serif'] = ['Times New Roman']
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['axes.unicode_minus'] = False

    os.makedirs(save_dir, exist_ok=True)

    metric = metric.lower()
    if metric not in ['f1', 'accuracy', 'acc']:
        raise ValueError("The metric only supports 'f1' or 'accuracy'")

    if metric == 'acc':
        metric = 'accuracy'

    if name is None:
        name = f'{model}_{metric}_boxplot.png'

    base_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b',
        '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8'
    ]

    def is_ours_model(model_name):
        return ('Ours' in model_name) or ('PACENet' in model_name)

    def display_model_name(model_name):
        if is_ours_model(model_name):
            return 'PACENet'
        return model_name

    def prob_to_pred(y_prob):
        y_prob = np.asarray(y_prob)

        if n_classes == 2:
            if y_prob.ndim == 1:
                y_prob_pos = y_prob
            elif y_prob.ndim == 2 and y_prob.shape[1] == 2:
                y_prob_pos = y_prob[:, 1]
            elif y_prob.ndim == 2 and y_prob.shape[1] == 1:
                y_prob_pos = y_prob[:, 0]
            else:
                raise ValueError(f'Two-class y_prob dimension anomaly: {y_prob.shape}')

            y_pred = (y_prob_pos >= threshold).astype(int)

        else:
            if y_prob.ndim != 2:
                raise ValueError(f'The y_prob for multi-class classification should be [N, C], but currently it is: {y_prob.shape}')
            y_pred = np.argmax(y_prob, axis=1)

        return y_pred

    def compute_metric(y_true, y_pred):
        if metric == 'f1':
            return f1_score(
                y_true,
                y_pred,
                average=average,
                zero_division=0
            )

        if metric == 'accuracy':
            return accuracy_score(y_true, y_pred)

    plot_data = []

    color_idx = 0

    for model_name, data in benchmark_results_dict.items():
        if model_name == 'SVM' or model_name == 'XGBoost':
            continue
        y_true_key = f'{model}_y_true_list'
        y_prob_key = f'{model}_y_prob_list'

        y_true_list = data[y_true_key]
        y_prob_list = data[y_prob_key]

        if len(y_true_list) != len(y_prob_list):
            raise ValueError(
                f'{model_name} 的 {y_true_key} 与 {y_prob_key} 折数不一致：'
                f'{len(y_true_list)} vs {len(y_prob_list)}'
            )

        fold_scores = []

        for fold_idx, (y_true, y_prob) in enumerate(zip(y_true_list, y_prob_list)):
            y_true = np.asarray(y_true)
            y_pred = prob_to_pred(y_prob)

            if len(y_true) != len(y_pred):
                raise ValueError(
                    f'In the {model_name} fold {fold_idx}, the lengths of y_true and y_pred are not consistent:'
                    f'{len(y_true)} vs {len(y_pred)}'
                )

            score = compute_metric(y_true, y_pred)
            fold_scores.append(score)

        if len(fold_scores) == 0:
            continue

        is_ours = is_ours_model(model_name)
        color = '#d62728' if is_ours else base_colors[color_idx % len(base_colors)]

        if not is_ours:
            color_idx += 1

        plot_data.append({
            'raw_name': model_name,
            'name': display_model_name(model_name),
            'scores': fold_scores,
            'mean': float(np.mean(fold_scores)),
            'std': float(np.std(fold_scores)),
            'color': color,
            'is_ours': is_ours
        })

    if len(plot_data) == 0:
        raise ValueError(
            f'There are no available data for plotting. Please check if {model}_y_true_list and {model}_y_prob_list exist in the benchmark_results_dict.')

    plot_data.sort(key=lambda x: (not x['is_ours'], -x['mean']))

    values_to_plot = [item['scores'] for item in plot_data]
    colors_to_plot = [item['color'] for item in plot_data]

    plt.figure(figsize=(10, 8))

    bp = plt.boxplot(
        values_to_plot,
        patch_artist=True,
        showmeans=False,
        showfliers=False,
        widths=0.55,
        meanprops={
            'marker': 'o',
            'markerfacecolor': 'white',
            'markeredgecolor': 'black',
            'markersize': 4
        },
        medianprops={
            'color': 'black',
            'linewidth': 1.5
        },
        boxprops={
            'linewidth': 1.2,
            'edgecolor': 'black'
        },
        whiskerprops={
            'linewidth': 1.1,
            'color': 'black'
        },
        capprops={
            'linewidth': 1.1,
            'color': 'black'
        },
        flierprops={
            'marker': 'o',
            'markersize': 3,
            'markerfacecolor': 'gray',
            'markeredgecolor': 'gray',
            'alpha': 0.6
        }
    )

    for patch, color in zip(bp['boxes'], colors_to_plot):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    metric_name = 'Macro-Averaged F1-score' if metric == 'f1' else 'Accuracy'

    plt.xlabel('Models', fontsize=16, weight='bold')
    plt.ylabel(metric_name, fontsize=16, weight='bold')

    plt.xticks([])
    plt.yticks(fontsize=12)

    plt.grid(True, axis='y', linestyle='--', alpha=0.6)

    all_values = np.concatenate([np.asarray(v) for v in values_to_plot])
    y_min = max(0.0, np.min(all_values) - 0.05)
    y_max = min(1.02, np.max(all_values) + 0.05)

    if y_min > 0.55:
        y_min = max(0.50, y_min)

    plt.ylim(y_min, y_max)

    legend_handles = []

    for item in plot_data:
        legend_handles.append(
            Patch(
                facecolor=item['color'],
                edgecolor='black',
                label=f"{item['name']} ({item['mean']:.4f})",
                alpha=0.85
            )
        )

    legend = plt.legend(
        handles=legend_handles,
        loc='lower left',
        fontsize=16,
        framealpha=0.95,
        edgecolor='black',
        fancybox=True
    )

    for text, item in zip(legend.get_texts(), plot_data):
        if 'PACENet' in item['name']:
            text.set_fontweight('bold')

    plt.tight_layout()

    save_path = os.path.join(save_dir, name)
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()

    return {
        item['raw_name']: {
            'scores': item['scores'],
            'mean': item['mean'],
            'std': item['std']
        }
        for item in plot_data
    }

def plot_ablation_metric_boxplots(results_dict, n_classes, save_dir, model='test', metric='f1', threshold=0.5, average='macro',
                                  experiment_group='architecture', show_points=False):

    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['font.serif'] = ['Times New Roman']
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['axes.unicode_minus'] = False

    os.makedirs(save_dir, exist_ok=True)

    supported_metrics = ['f1', 'accuracy', 'auroc', 'auprc']
    if metric not in supported_metrics:
        raise ValueError(f"metric only supports {supported_metrics}")

    name = f'{model}_{metric}_{experiment_group}_ablation_boxplot.png'

    architecture_order = [
        'Full PACENet',
        'w/o UFE',
        'w/o CFE',
        'w/o SFE',
        'w/o KFE'
    ]

    augmentation_order = [
        'Aug_Full PACENet',
        'Aug_w/o Data Augmentation',
        'Aug_w/o Jitter',
        'Aug_w/o Scaling',
        'Aug_w/o Magnitude Warp',
        'Aug_w/o Time Warp',
        'Aug_w/o Random Bias',
        'Aug_w/o Crosstalk'
    ]

    if experiment_group == 'architecture':
        selected_names = [m for m in architecture_order if m in results_dict]
    elif experiment_group == 'augmentation':
        selected_names = [m for m in augmentation_order if m in results_dict]
    elif experiment_group == 'all':
        selected_names = list(results_dict.keys())
    else:
        raise ValueError("experiment_group must be 'architecture', 'augmentation', or 'all'")

    if len(selected_names) == 0:
        raise ValueError(f"No valid entries found for experiment_group='{experiment_group}'")

    def is_ours_model(model_name):
        return ('Ours' in model_name) or ('PACENet' in model_name)

    def display_model_name(model_name):
        name = model_name.replace('Aug_', '')
        if name in ['Full PACENet', 'PACENet(Ours)', 'PACE-Net(Ours)']:
            return 'Full PACENet'
        return name

    def x_stick__name(model_name, experiment_group):
        name = model_name.replace('Aug_', '')
        if experiment_group == 'augmentation':
            if name in ['Full PACENet', 'PACENet(Ours)', 'PACE-Net(Ours)']:
                return 'Full\nPACENet'
            if 'w/o' in name:
                return 'w/o\n' + name[4:]
        return name

    def prob_to_pred(y_prob):
        y_prob = np.asarray(y_prob)

        if n_classes == 2:
            if y_prob.ndim == 1:
                y_prob_pos = y_prob
            elif y_prob.ndim == 2 and y_prob.shape[1] == 2:
                y_prob_pos = y_prob[:, 1]
            elif y_prob.ndim == 2 and y_prob.shape[1] == 1:
                y_prob_pos = y_prob[:, 0]
            else:
                raise ValueError(f'Binary y_prob shape error: {y_prob.shape}')

            y_pred = (y_prob_pos >= threshold).astype(int)

        else:
            if y_prob.ndim != 2:
                raise ValueError(f'Multiclass y_prob should be [N, C], got {y_prob.shape}')
            y_pred = np.argmax(y_prob, axis=1)

        return y_pred

    def compute_metric(y_true, y_prob):
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)
        y_pred = prob_to_pred(y_prob)

        if len(y_true) != len(y_pred):
            raise ValueError(f'y_true and y_pred length mismatch: {len(y_true)} vs {len(y_pred)}')

        if metric == 'f1':
            return f1_score(y_true, y_pred, average=average, zero_division=0)

        if metric == 'accuracy':
            return accuracy_score(y_true, y_pred)

        if metric == 'auroc':
            if n_classes == 2:
                if y_prob.ndim == 1:
                    y_score = y_prob
                elif y_prob.ndim == 2 and y_prob.shape[1] == 2:
                    y_score = y_prob[:, 1]
                else:
                    y_score = y_prob[:, 0]
                return roc_auc_score(y_true, y_score)

            return roc_auc_score(
                y_true,
                y_prob,
                multi_class='ovr',
                average=average
            )

        if metric == 'auprc':
            if n_classes == 2:
                if y_prob.ndim == 1:
                    y_score = y_prob
                elif y_prob.ndim == 2 and y_prob.shape[1] == 2:
                    y_score = y_prob[:, 1]
                else:
                    y_score = y_prob[:, 0]
                return average_precision_score(y_true, y_score)

            y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
            return average_precision_score(
                y_true_bin,
                y_prob,
                average=average
            )

    base_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
        '#17becf', '#aec7e8'
    ]

    plot_data = []
    color_idx = 0

    y_true_key = f'{model}_y_true_list'
    y_prob_key = f'{model}_y_prob_list'

    for model_name in selected_names:
        data = results_dict[model_name]

        if y_true_key not in data or y_prob_key not in data:
            print(f'[Warning] {model_name} missing {y_true_key} or {y_prob_key}, skipped.')
            continue

        y_true_list = data[y_true_key]
        y_prob_list = data[y_prob_key]

        if len(y_true_list) != len(y_prob_list):
            raise ValueError(
                f'{model_name}: fold number mismatch: '
                f'{len(y_true_list)} vs {len(y_prob_list)}'
            )

        fold_scores = []
        for fold_idx, (y_true, y_prob) in enumerate(zip(y_true_list, y_prob_list)):
            try:
                score = compute_metric(y_true, y_prob)
                fold_scores.append(score)
            except Exception as e:
                print(f'[Warning] {model_name} fold {fold_idx} failed: {e}')

        if len(fold_scores) == 0:
            continue

        is_ours = is_ours_model(model_name) or ('Full PACENet' in model_name)
        color = '#d62728' if is_ours else base_colors[color_idx % len(base_colors)]

        if not is_ours:
            color_idx += 1

        plot_data.append({
            'raw_name': model_name,
            'name': display_model_name(model_name),
            'xtick_labels': x_stick__name(model_name, experiment_group),
            'scores': fold_scores,
            'mean': float(np.mean(fold_scores)),
            'std': float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0,
            'color': color,
            'is_ours': is_ours
        })

    if len(plot_data) == 0:
        raise ValueError('No valid data for plotting.')

    order_map = {name: i for i, name in enumerate(selected_names)}
    plot_data.sort(key=lambda x: order_map.get(x['raw_name'], 999))

    values_to_plot = [item['scores'] for item in plot_data]
    colors_to_plot = [item['color'] for item in plot_data]

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    bp = ax.boxplot(
        values_to_plot,
        patch_artist=True,
        showmeans=False,
        showfliers=False,
        widths=0.55,
        medianprops={
            'color': 'black',
            'linewidth': 1.5
        },
        boxprops={
            'linewidth': 1.2,
            'edgecolor': 'black'
        },
        whiskerprops={
            'linewidth': 1.1,
            'color': 'black'
        },
        capprops={
            'linewidth': 1.1,
            'color': 'black'
        }
    )

    for patch, color in zip(bp['boxes'], colors_to_plot):
        patch.set_facecolor(color)
        patch.set_alpha(0.80)

    # Overlay fold-level points
    if show_points:
        rng = np.random.default_rng(42)
        for i, item in enumerate(plot_data, start=1):
            scores = np.asarray(item['scores'])
            jitter = rng.uniform(-0.07, 0.07, size=len(scores))
            ax.scatter(
                np.full(len(scores), i) + jitter,
                scores,
                s=32,
                color='white',
                edgecolor='black',
                linewidth=0.7,
                zorder=3,
                alpha=0.95
            )


    metric_name_map = {
        'f1': 'Macro-Averaged F1-score',
        'accuracy': 'Accuracy',
        'auroc': 'Macor-Averaged AUROC',
        'auprc': 'Macor-Averaged AUPRC'
    }
    metric_name = metric_name_map[metric]

    ax.set_xlabel('Model', fontsize=18, weight='bold')
    ax.set_ylabel(metric_name, fontsize=18, weight='bold')

    ax.set_xticks(np.arange(1, len(plot_data) + 1))
    ax.set_xticklabels([item['xtick_labels'] for item in plot_data], fontsize=16)

    for label, item in zip(ax.get_xticklabels(), plot_data):
        if item['is_ours']:
            label.set_fontweight('bold')

    ax.tick_params(axis='y', labelsize=16)
    ax.grid(True, axis='y', linestyle='--', alpha=0.45)

    # Adaptive y-axis range
    all_values = np.concatenate([np.asarray(v) for v in values_to_plot])
    y_min = max(0.0, np.min(all_values) - 0.04)
    y_max = min(1.02, np.max(all_values) + 0.04)
    if y_min > 0.50:
        y_min = max(0.50, y_min)
    ax.set_ylim(y_min, y_max)

    # Legend
    legend_handles = [
        Patch(
            facecolor=item['color'],
            edgecolor='black',
            label=f"{item['name']} ({item['mean']:.4f})",
            alpha=0.80
        )
        for item in plot_data
    ]

    legend = ax.legend(
        handles=legend_handles,
        loc='lower left',
        fontsize=18,
        framealpha=0.95,
        edgecolor='black',
        fancybox=True
    )

    for text, item in zip(legend.get_texts(), plot_data):
        if 'PACENet' in item['name']:
            text.set_fontweight('bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    save_path = os.path.join(save_dir, name)
    plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()

    return {
        item['raw_name']: {
            'display_name': item['name'],
            'scores': item['scores'],
            'mean': item['mean'],
            'std': item['std']
        }
        for item in plot_data
    }

def plot_paired_delta_metric(results_dict, n_classes, save_dir, model='test', metric='f1', experiment_group='architecture',
                             threshold=0.5, average='macro', multiply_by_100=True, ci_level=0.95, show_fold_points=False,
                             annotate_mean=True, figsize=[10, 8], dpi=600,):
    try:
        from scipy.stats import t
        HAS_SCIPY = True
    except Exception:
        HAS_SCIPY = False

    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['font.serif'] = ['Times New Roman']
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['axes.unicode_minus'] = False

    os.makedirs(save_dir, exist_ok=True)

    name = f'{model}_{metric}_{experiment_group}_paired_delta_vs_full_pacenet.png'

    def is_full_model(model_name):
        compact = model_name.lower().replace('-', '').replace('_', '').replace(' ', '')
        return (
            'fullpacenet' in compact
            or 'pacenetours' in compact
            or compact == 'pacenet'
            or (
                'pacenet' in compact
                and 'wo' not in compact
                and 'without' not in compact
                and 'w/o' not in model_name.lower()
            )
        )

    def display_model_name(model_name):
        name_ = model_name.replace('Aug_', '')
        if is_full_model(name_):
            return 'Full PACENet'
        return name_

    def x_stick__name(model_name, experiment_group):
        name = model_name.replace('Aug_', '')
        if experiment_group == 'augmentation':
            if name in ['Full PACENet', 'PACENet(Ours)', 'PACE-Net(Ours)']:
                return 'Full\nPACENet'
            if 'w/o' in name:
                return 'w/o\n' + name[4:]
        return name

    def prob_to_pred(y_prob):
        y_prob = np.asarray(y_prob)

        if n_classes == 2:
            if y_prob.ndim == 1:
                y_prob_pos = y_prob
            elif y_prob.ndim == 2 and y_prob.shape[1] == 2:
                y_prob_pos = y_prob[:, 1]
            elif y_prob.ndim == 2 and y_prob.shape[1] == 1:
                y_prob_pos = y_prob[:, 0]
            else:
                raise ValueError(f'Binary y_prob shape error: {y_prob.shape}')

            y_pred = (y_prob_pos >= threshold).astype(int)

        else:
            if y_prob.ndim != 2:
                raise ValueError(f'Multiclass y_prob should be [N, C], got {y_prob.shape}')
            y_pred = np.argmax(y_prob, axis=1)

        return y_pred

    def compute_metric(y_true, y_prob):
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)
        y_pred = prob_to_pred(y_prob)

        if len(y_true) != len(y_pred):
            raise ValueError(
                f'y_true and y_pred length mismatch: {len(y_true)} vs {len(y_pred)}'
            )

        if metric == 'f1':
            return f1_score(y_true, y_pred, average=average, zero_division=0)

        if metric == 'accuracy':
            return accuracy_score(y_true, y_pred)

        if metric == 'auroc':
            if n_classes == 2:
                if y_prob.ndim == 1:
                    y_score = y_prob
                elif y_prob.ndim == 2 and y_prob.shape[1] == 2:
                    y_score = y_prob[:, 1]
                else:
                    y_score = y_prob[:, 0]
                return roc_auc_score(y_true, y_score)

            return roc_auc_score(
                y_true,
                y_prob,
                multi_class='ovr',
                average=average
            )

        if metric == 'auprc':
            if n_classes == 2:
                if y_prob.ndim == 1:
                    y_score = y_prob
                elif y_prob.ndim == 2 and y_prob.shape[1] == 2:
                    y_score = y_prob[:, 1]
                else:
                    y_score = y_prob[:, 0]
                return average_precision_score(y_true, y_score)

            y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
            return average_precision_score(
                y_true_bin,
                y_prob,
                average=average
            )

    def get_fold_scores(model_name):
        y_true_key = f'{model}_y_true_list'
        y_prob_key = f'{model}_y_prob_list'

        if model_name not in results_dict:
            raise ValueError(f"Model '{model_name}' not found in pkl.")

        data = results_dict[model_name]

        if y_true_key not in data or y_prob_key not in data:
            raise ValueError(
                f"{model_name} missing {y_true_key} or {y_prob_key}."
            )

        y_true_list = data[y_true_key]
        y_prob_list = data[y_prob_key]

        if len(y_true_list) != len(y_prob_list):
            raise ValueError(
                f"{model_name}: fold number mismatch: "
                f"{len(y_true_list)} vs {len(y_prob_list)}"
            )

        scores = []
        for fold_idx, (y_true, y_prob) in enumerate(zip(y_true_list, y_prob_list)):
            try:
                score = compute_metric(y_true, y_prob)
            except Exception as e:
                print(f"[Warning] {model_name} fold {fold_idx} failed: {e}")
                score = np.nan
            scores.append(score)

        return np.asarray(scores, dtype=float)

    def mean_ci(values, ci_level=0.95):
        values = np.asarray(values, dtype=float)
        values = values[~np.isnan(values)]

        n = len(values)
        mean = float(np.mean(values))

        if n <= 1:
            return mean, 0.0, n

        sd = float(np.std(values, ddof=1))
        sem = sd / np.sqrt(n)

        if HAS_SCIPY:
            alpha = 1 - ci_level
            tcrit = t.ppf(1 - alpha / 2, df=n - 1)
        else:
            tcrit = 1.96

        ci = float(tcrit * sem)
        return mean, ci, n

    all_names = list(results_dict.keys())

    baseline_candidates = [m for m in all_names if is_full_model(m)]
    baseline_name = baseline_candidates[0]

    architecture_order = [
        'w/o UFE',
        'w/o CFE',
        'w/o SFE',
        'w/o KFE'
    ]

    augmentation_order = [
        'w/o Data Augmentation',
        'w/o Jitter',
        'w/o Scaling',
        'w/o Magnitude Warp',
        'w/o Time Warp',
        'w/o Random Bias',
        'w/o Crosstalk'
    ]

    augmentation_order_aug = [
        'Aug_w/o Data Augmentation',
        'Aug_w/o Jitter',
        'Aug_w/o Scaling',
        'Aug_w/o Magnitude Warp',
        'Aug_w/o Time Warp',
        'Aug_w/o Random Bias',
        'Aug_w/o Crosstalk'
    ]

    if experiment_group == 'architecture':
        ablation_names = [m for m in architecture_order if m in results_dict]

    elif experiment_group == 'augmentation':
        ablation_names = [m for m in augmentation_order if m in results_dict]

        if len(ablation_names) == 0:
            ablation_names = [m for m in augmentation_order_aug if m in results_dict]

    elif experiment_group == 'all':
        ablation_names = [m for m in all_names if m != baseline_name]

    else:
        raise ValueError(
            "experiment_group must be 'architecture', 'augmentation', or 'all'"
        )

    if len(ablation_names) == 0:
        raise ValueError(
            f"No ablation models found for experiment_group='{experiment_group}'. "
            "Please check model names in the pkl file or pass model_order manually."
        )

    baseline_scores = get_fold_scores(baseline_name)

    plot_data = []

    for ablation_name in ablation_names:
        ablation_scores = get_fold_scores(ablation_name)

        n_pair = min(len(baseline_scores), len(ablation_scores))
        if len(baseline_scores) != len(ablation_scores):
            print(
                f"[Warning] {ablation_name}: fold number differs from baseline. "
                f"Using first {n_pair} paired folds."
            )

        base = baseline_scores[:n_pair]
        abl = ablation_scores[:n_pair]

        valid_mask = (~np.isnan(base)) & (~np.isnan(abl))
        delta = abl[valid_mask] - base[valid_mask]

        if multiply_by_100:
            delta = delta * 100

        mean_delta, ci_delta, n_valid = mean_ci(delta, ci_level=ci_level)

        plot_data.append({
            'raw_name': ablation_name,
            'name': display_model_name(ablation_name),
            'xtick_labels': x_stick__name(ablation_name, experiment_group),
            'delta': delta,
            'mean_delta': mean_delta,
            'ci_delta': ci_delta,
            'n': n_valid
        })

    metric_name_map = {
        'f1': 'Macro-Averaged F1-score',
        'accuracy': 'Accuracy',
        'auroc': 'Macro-Averaged AUROC',
        'auprc': 'Macro-Averaged AUPRC'
    }
    metric_name = metric_name_map[metric]

    group_title_map = {
        'architecture': 'Architecture Ablation',
        'augmentation': 'Augmentation Ablation',
        'all': 'ablation'
    }

    if figsize is None:
        figsize = (9, max(4.8, 0.75 * len(plot_data) + 2.0))

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    x_positions = np.arange(len(plot_data))

    # 0 reference line
    ax.axhline(0, color='black', linewidth=1.2, linestyle='-', alpha=0.9)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2']
    rng = np.random.default_rng(42)

    for idx, (item, x) in enumerate(zip(plot_data, x_positions)):
        color = colors[idx % len(colors)]

        if show_fold_points:
            jitter = rng.uniform(-0.08, 0.08, size=len(item['delta']))
            ax.scatter(
                np.full(len(item['delta']), x) + jitter,
                item['delta'],
                s=42,
                color=color,
                alpha=0.65,
                edgecolor='white',
                linewidth=0.6,
                zorder=2
            )

        ax.errorbar(
            x,
            item['mean_delta'],
            yerr=item['ci_delta'],
            fmt='D',
            markersize=9,
            color=color,
            ecolor=color,
            elinewidth=2.2,
            capsize=5,
            capthick=1.6,
            zorder=4
        )

        if annotate_mean:
            unit = " pp" if multiply_by_100 else ""

            if idx < len(plot_data):
                    x_text = x + 0.05
                    ha = 'left'

            y_text = item['mean_delta'] - 0.06

            ax.text(
                x_text,
                y_text,
                f"{item['mean_delta']:+.2f}{unit}",
                ha=ha,
                va='center',
                fontsize=18,
                color=color,
                fontweight='bold'
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [item['xtick_labels'] for item in plot_data],
        fontsize=16
    )

    ylabel = f"Δ{metric_name} vs Full PACENet"
    if multiply_by_100:
        ylabel += " (percentage points)"
    ax.set_ylabel(ylabel, fontsize=18, fontweight='bold')
    ax.set_xlabel('Model', fontsize=18, fontweight='bold')

    ax.grid(axis='y', linestyle='--', alpha=0.45)
    ax.grid(axis='x', linestyle='-', alpha=0.12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    all_delta = np.concatenate([item['delta'] for item in plot_data if len(item['delta']) > 0])
    y_min = min(np.min(all_delta), min(item['mean_delta'] - item['ci_delta'] for item in plot_data))
    y_max = max(np.max(all_delta), max(item['mean_delta'] + item['ci_delta'] for item in plot_data))

    pad = max(0.8 if multiply_by_100 else 0.008, 0.15 * (y_max - y_min + 1e-8))
    ax.set_ylim(y_min - pad, y_max + pad)

    plt.tight_layout()

    save_path = os.path.join(save_dir, name)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()

    return {
        item['raw_name']: {
            'display_name': item['name'],
            'delta': item['delta'],
            'mean_delta': item['mean_delta'],
            'ci_delta': item['ci_delta'],
            'n': item['n']
        }
        for item in plot_data
    }

def plot_raw_signal_attention(plot_data, save_dir, kernel_size=40, stride=3):

    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['font.serif'] = ['Times New Roman']
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['axes.unicode_minus'] = False

    def map_attention_to_signal(attn_1d, raw_signal):
        _, time_steps = raw_signal.shape
        seq_len = len(attn_1d)
        attn_upsampled = np.zeros(time_steps)

        sigma_window = kernel_size / 6.0
        x = np.arange(kernel_size)
        gaussian_window = np.exp(-0.5 * ((x - kernel_size / 2) / sigma_window) ** 2)

        for i in range(seq_len):
            start_idx = i * stride
            end_idx = min(start_idx + kernel_size, time_steps)
            actual_len = end_idx - start_idx
            weighted_attn = attn_1d[i] * gaussian_window[:actual_len]
            attn_upsampled[start_idx:end_idx] = np.maximum(
                attn_upsampled[start_idx:end_idx],
                weighted_attn
            )

        attn_norm = (attn_upsampled - np.min(attn_upsampled)) / (np.max(attn_upsampled) - np.min(attn_upsampled) + 1e-8)
        attn_sharpened = attn_norm ** 3
        attn_smooth = gaussian_filter1d(attn_sharpened, sigma=4)

        attn_smooth = (attn_smooth - np.min(attn_smooth)) / (np.max(attn_smooth) - np.min(attn_smooth) + 1e-8)

        return attn_smooth

    channel_names = ['L_AA_Angle', 'L_IER_Angle', 'L_FE_Angle', 'L_AP_Translation', 'L_PD_Translation', 'L_ML_Translation',
                          'R_AA_Angle', 'R_IER_Angle', 'R_FE_Angle', 'R_AP_Translation', 'R_PD_Translation', 'R_ML_Translation']

    for item in plot_data['attn_weight']:
        raw_signal, attn_left_1d, attn_right_1d, name = item['raw_signal'], item['attn_left_1d'], item['attn_right_1d'], item['image_filename']
        _, time_steps = raw_signal.shape

        attn_up_left = map_attention_to_signal(attn_left_1d, raw_signal)
        attn_up_right = map_attention_to_signal(attn_right_1d, raw_signal)
        cmap = plt.get_cmap('jet')

        fig, axes = plt.subplots(6, 2, figsize=(16, 12), sharex=True)
        x_time = np.arange(time_steps)

        for row in range(6):
            ax_l = axes[row, 0]
            sig_l = raw_signal[row]
            ymin_l, ymax_l = np.min(sig_l) - 0.5, np.max(sig_l) + 0.5

            ax_l.imshow(attn_up_left[np.newaxis, :], cmap=cmap, aspect='auto',
                        extent=[0, time_steps, ymin_l, ymax_l], alpha=0.5, vmin=0, vmax=1,
                        interpolation='bicubic')
            ax_l.plot(x_time, sig_l, color='black', linewidth=1.2)
            ax_l.set_ylabel(channel_names[row], fontsize=11, weight='bold')
            ax_l.set_ylim(ymin_l, ymax_l)
            ax_l.set_xlim(0, time_steps)
            ax_l.set_yticks([])
            ax_l.spines['top'].set_visible(False)
            ax_l.spines['right'].set_visible(False)

            ax_r = axes[row, 1]
            sig_r = raw_signal[row + 6]
            ymin_r, ymax_r = np.min(sig_r) - 0.5, np.max(sig_r) + 0.5

            ax_r.imshow(attn_up_right[np.newaxis, :], cmap=cmap, aspect='auto',
                        extent=[0, time_steps, ymin_r, ymax_r], alpha=0.5, vmin=0, vmax=1,
                        interpolation='bicubic')
            ax_r.plot(x_time, sig_r, color='black', linewidth=1.2)
            ax_r.set_ylabel(channel_names[row + 6], fontsize=11, weight='bold')
            ax_r.set_ylim(ymin_r, ymax_r)
            ax_r.set_xlim(0, time_steps)
            ax_r.set_yticks([])
            ax_r.spines['top'].set_visible(False)
            ax_r.spines['right'].set_visible(False)

        axes[0, 0].set_title('Left Leg', fontsize=14, weight='bold', pad=10)
        axes[0, 1].set_title('Right Leg', fontsize=14, weight='bold', pad=10)
        axes[-1, 0].set_xlabel('Time Steps', fontsize=12, weight='bold')
        axes[-1, 1].set_xlabel('Time Steps', fontsize=12, weight='bold')

        plt.tight_layout()
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, name), dpi=600, bbox_inches='tight')
        plt.close()

def plot_acld_affected_vs_unaffected_attention(plot_data, save_dir, kernel_size=40, stride=3):
    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['font.serif'] = ['Times New Roman']
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['axes.unicode_minus'] = False

    def map_attention_to_signal(attn_1d, raw_signal):
        _, time_steps = raw_signal.shape
        seq_len = len(attn_1d)
        attn_upsampled = np.zeros(time_steps)

        sigma_window = kernel_size / 6.0
        x = np.arange(kernel_size)
        gaussian_window = np.exp(-0.5 * ((x - kernel_size / 2) / sigma_window) ** 2)

        for i in range(seq_len):
            start_idx = i * stride
            end_idx = min(start_idx + kernel_size, time_steps)
            actual_len = end_idx - start_idx
            weighted_attn = attn_1d[i] * gaussian_window[:actual_len]
            attn_upsampled[start_idx:end_idx] = np.maximum(
                attn_upsampled[start_idx:end_idx],
                weighted_attn
            )

        attn_norm = (attn_upsampled - np.min(attn_upsampled)) / (np.max(attn_upsampled) - np.min(attn_upsampled) + 1e-8)
        attn_sharpened = attn_norm ** 3
        attn_smooth = gaussian_filter1d(attn_sharpened, sigma=4)

        attn_smooth = (attn_smooth - np.min(attn_smooth)) / (np.max(attn_smooth) - np.min(attn_smooth) + 1e-8)

        return attn_smooth

    channel_names = ['L_AA_Angle', 'L_IER_Angle', 'L_FE_Angle', 'L_AP_Translation', 'L_PD_Translation',
                     'L_ML_Translation',
                     'R_AA_Angle', 'R_IER_Angle', 'R_FE_Angle', 'R_AP_Translation', 'R_PD_Translation',
                     'R_ML_Translation']

    for item in plot_data['attn_weight']:
        if item['label_name'] != 'ACLD':
            continue
        raw_signal = item['raw_signal']
        _, time_steps = raw_signal.shape
        acld_side = item['acld_side']
        orig_id = item['orig_id']

        name = f'ACLD_affected_vs_unaffected_attention_ID_{orig_id}.png'

        attn_left = map_attention_to_signal(item['attn_left_1d'], raw_signal)
        attn_right = map_attention_to_signal(item['attn_right_1d'], raw_signal)

        if acld_side == 'L':
            affected_signal = raw_signal[:6]
            unaffected_signal = raw_signal[6:]
            affected_attn = attn_left
            unaffected_attn = attn_right
        else:
            affected_signal = raw_signal[6:]
            unaffected_signal = raw_signal[:6]
            affected_attn = attn_right
            unaffected_attn = attn_left

        cmap = plt.get_cmap('jet')

        fig, axes = plt.subplots(6, 2, figsize=(16, 12), sharex=True)
        x_time = np.arange(time_steps)

        for row in range(6):
            ax_l = axes[row, 0]
            sig_l = affected_signal[row]
            ymin_l, ymax_l = np.min(sig_l) - 0.5, np.max(sig_l) + 0.5

            ax_l.imshow(affected_attn[np.newaxis, :], cmap=cmap, aspect='auto',
                        extent=[0, time_steps, ymin_l, ymax_l], alpha=0.5, vmin=0, vmax=1,
                        interpolation='bicubic')
            ax_l.plot(x_time, sig_l, color='black', linewidth=1.2)
            ax_l.set_ylabel(channel_names[row], fontsize=11, weight='bold')
            ax_l.set_ylim(ymin_l, ymax_l)
            ax_l.set_xlim(0, time_steps)
            ax_l.set_yticks([])
            ax_l.spines['top'].set_visible(False)
            ax_l.spines['right'].set_visible(False)

            ax_r = axes[row, 1]
            sig_r = unaffected_signal[row]
            ymin_r, ymax_r = np.min(sig_r) - 0.5, np.max(sig_r) + 0.5

            ax_r.imshow(unaffected_attn[np.newaxis, :], cmap=cmap, aspect='auto',
                        extent=[0, time_steps, ymin_r, ymax_r], alpha=0.5, vmin=0, vmax=1,
                        interpolation='bicubic')
            ax_r.plot(x_time, sig_r, color='black', linewidth=1.2)
            ax_r.set_ylabel(channel_names[row + 6], fontsize=11, weight='bold')
            ax_r.set_ylim(ymin_r, ymax_r)
            ax_r.set_xlim(0, time_steps)
            ax_r.set_yticks([])
            ax_r.spines['top'].set_visible(False)
            ax_r.spines['right'].set_visible(False)

        axes[0, 0].set_title('Affected Leg', fontsize=14, weight='bold', pad=10)
        axes[0, 1].set_title('Unaffected Leg', fontsize=14, weight='bold', pad=10)
        axes[-1, 0].set_xlabel('Time Steps', fontsize=12, weight='bold')
        axes[-1, 1].set_xlabel('Time Steps', fontsize=12, weight='bold')

        plt.tight_layout()
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, name), dpi=600, bbox_inches='tight')
        plt.close()


def plot_spectral_weights(plot_data, save_dir, in_channels=12, seq_len=600, fs=60, name='spectral_weights.png'):

    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['font.serif'] = ['Times New Roman']
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['axes.unicode_minus'] = False

    logger = logging.getLogger(__name__)

    weights_list = plot_data['spectral_weights']
    avg_weights = np.mean(np.abs(weights_list), axis=0)
    importance_flat = np.mean(avg_weights, axis=0)
    fft_dim = seq_len // 2 + 1
    importance_matrix = importance_flat.reshape(in_channels, fft_dim)
    freq_importance = np.mean(importance_matrix, axis=0)

    freqs = np.fft.rfftfreq(seq_len, d=1 / fs)
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(freqs, freq_importance, marker='o', color='#d62728', linewidth=2, markersize=5, zorder=4)
    ax.fill_between(freqs, 0, freq_importance, color='#d62728', alpha=0.3, zorder=1)

    top_k = 10
    top_indices = np.argsort(freq_importance)[-top_k:]
    top_indices = np.sort(top_indices)

    for idx in top_indices:
        x_val = freqs[idx]
        y_val = freq_importance[idx]
        ax.vlines(x=x_val, ymin=0, ymax=y_val, colors='#666666', linestyles='--', linewidth=1.5, alpha=0.8, zorder=2)

    ax.set_xlabel('Frequency (Hz)', fontsize=14, weight='bold')
    ax.set_ylabel('Mean Absolute Weight', fontsize=14, weight='bold')

    max_y = np.max(freq_importance)
    ax.set_ylim(bottom=0, top=max_y * 1.05)
    ax.set_xlim(left=0, right=min(30, np.max(freqs)))

    ax.grid(True, linestyle='--', alpha=0.6, zorder=0)

    plt.setp(ax.get_xticklabels(), fontsize=12, fontweight='bold')
    plt.setp(ax.get_yticklabels(), fontsize=12, fontweight='bold')

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, name), dpi=600, bbox_inches='tight')
    plt.close()

    logger.info(f"The visualization graph of frequency domain feature weights has been saved to: {os.path.join(save_dir, name)}")

def plot_kfold_dca_curve(y_true_list, y_prob_list, task_name, save_dir, name='kfold_dca_curve.png', positive_classes=(1, 2),
    model_label='PACENet', threshold_min=0.01, threshold_max=0.80, n_thresholds=100, show_std=True):
    thresholds = np.linspace(threshold_min, threshold_max, n_thresholds)

    k_folds = len(y_true_list)
    model_net_benefits = []
    treat_all_net_benefits = []

    for i in range(k_folds):
        y_true = np.asarray(y_true_list[i]).astype(int)
        y_prob = np.asarray(y_prob_list[i])

        y_true_bin = np.isin(y_true, list(positive_classes)).astype(int)

        if y_prob.ndim == 1:
            positive_score = y_prob

        elif y_prob.ndim == 2:
            if y_prob.shape[1] == 2:
                positive_score = y_prob[:, 1]

            else:
                positive_score = y_prob[:, list(positive_classes)].sum(axis=1)

        else:
            raise ValueError(f"Unsupported y_prob shape: {y_prob.shape}")

        n = len(y_true_bin)

        fold_model_nb = []
        fold_all_nb = []

        for pt in thresholds:
            preds = (positive_score >= pt).astype(int)

            tp = np.sum((preds == 1) & (y_true_bin == 1))
            fp = np.sum((preds == 1) & (y_true_bin == 0))

            nb_model = (tp / n) - (fp / n) * (pt / (1 - pt))
            fold_model_nb.append(nb_model)

            tp_all = np.sum(y_true_bin == 1)
            fp_all = np.sum(y_true_bin == 0)

            nb_all = (tp_all / n) - (fp_all / n) * (pt / (1 - pt))
            fold_all_nb.append(nb_all)

        model_net_benefits.append(fold_model_nb)
        treat_all_net_benefits.append(fold_all_nb)

    model_net_benefits = np.asarray(model_net_benefits)
    treat_all_net_benefits = np.asarray(treat_all_net_benefits)

    mean_model_nb = np.mean(model_net_benefits, axis=0)
    std_model_nb = np.std(model_net_benefits, axis=0)

    mean_all_nb = np.mean(treat_all_net_benefits, axis=0)

    plt.figure(figsize=(18, 11))

    plt.plot(
        thresholds,
        np.zeros_like(thresholds),
        color='black',
        linewidth=4.0,
        linestyle='--',
        label='Treat None'
    )

    plt.plot(
        thresholds,
        mean_all_nb,
        color='gray',
        linewidth=4.0,
        linestyle=':',
        label='Treat All'
    )

    plt.plot(
        thresholds,
        mean_model_nb,
        color='#d62728',
        linewidth=8.0,
        label=f'{model_label}'
    )

    if show_std:
        plt.fill_between(
            thresholds,
            mean_model_nb - std_model_nb,
            mean_model_nb + std_model_nb,
            color='#d62728',
            alpha=0.2
        )

    ymax = max(np.max(mean_model_nb), np.max(mean_all_nb), 0.0) + 0.05
    plt.ylim([-0.05, ymax])
    plt.xlim([threshold_min, threshold_max])

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.xlabel('Threshold Probability', fontsize=24, weight='bold')
    plt.ylabel('Net Benefit', fontsize=24, weight='bold')
    plt.legend(loc='lower left', fontsize=28)
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, name), dpi=600, bbox_inches='tight')
    plt.close()


def quantify_acld_affected_vs_unaffected_attention(plot_data, save_dir=None, only_acld=True, top_frac=0.10):
    def _normalize_attention(attn):
        attn = np.asarray(attn, dtype=np.float64)
        attn = np.maximum(attn, 0)
        return attn / (np.sum(attn) + 1e-12)

    def _attention_entropy(p):
        p = _normalize_attention(p)
        n = len(p)
        entropy = -np.sum(p * np.log(p + 1e-12))
        entropy = entropy / (np.log(n) + 1e-12)
        return entropy

    def _attention_gini(p):
        p = _normalize_attention(p)
        x = np.sort(p)
        n = len(x)
        if np.sum(x) <= 0:
            return 0.0
        gini = (2.0 * np.sum((np.arange(1, n + 1)) * x) / (n * np.sum(x))) - (n + 1) / n
        return gini

    def _attention_stats(attn, top_frac=0.10):
        p = _normalize_attention(attn)
        n = len(p)

        top_k = max(1, int(np.ceil(n * top_frac)))
        top_mass = np.sum(np.sort(p)[-top_k:])

        peak = np.max(p)
        peak_idx = int(np.argmax(p))
        peak_loc = peak_idx / max(1, n - 1)

        entropy = _attention_entropy(p)
        gini = _attention_gini(p)

        one_third = n // 3
        two_third = 2 * n // 3

        early_mass = np.sum(p[:one_third])
        middle_mass = np.sum(p[one_third:two_third])
        late_mass = np.sum(p[two_third:])

        return {
            'entropy': entropy,
            'gini': gini,
            'peak': peak,
            'peak_idx': peak_idx,
            'peak_loc': peak_loc,
            'top10_mass': top_mass,
            'early_mass': early_mass,
            'middle_mass': middle_mass,
            'late_mass': late_mass
        }

    records = []
    attn_weight = plot_data['attn_weight']

    for item in attn_weight:
        if only_acld and item.get('label_name') != 'ACLD':
            continue

        acld_side = item.get('acld_side')
        orig_id = item.get('orig_id')

        if acld_side not in ['L', 'R', 'left', 'right']:
            continue

        attn_left = _normalize_attention(item['attn_left_1d'])
        attn_right = _normalize_attention(item['attn_right_1d'])

        if acld_side in ['L', 'left']:
            affected_attn = attn_left
            unaffected_attn = attn_right
        else:
            affected_attn = attn_right
            unaffected_attn = attn_left

        affected_stats = _attention_stats(affected_attn, top_frac=top_frac)
        unaffected_stats = _attention_stats(unaffected_attn, top_frac=top_frac)

        eps = 1e-8

        record = {
            'orig_id': orig_id,
            'label_name': item.get('label_name'),
            'acld_side': acld_side,
        }

        for key in affected_stats.keys():
            record[f'affected_{key}'] = affected_stats[key]
            record[f'unaffected_{key}'] = unaffected_stats[key]

        for key in ['peak', 'top10_mass', 'gini', 'early_mass', 'middle_mass', 'late_mass']:
            aff = affected_stats[key]
            unaff = unaffected_stats[key]
            record[f'{key}_diff'] = aff - unaff
            record[f'{key}_ratio'] = aff / (unaff + eps)
            record[f'{key}_affected_share'] = aff / (aff + unaff + eps)
            record[f'{key}_dominance_index'] = (aff - unaff) / (aff + unaff + eps)

        aff_entropy = affected_stats['entropy']
        unaff_entropy = unaffected_stats['entropy']
        record['entropy_diff'] = aff_entropy - unaff_entropy
        record['entropy_ratio'] = aff_entropy / (unaff_entropy + eps)
        record['entropy_focus_dominance_index'] = (
            unaff_entropy - aff_entropy
        ) / (unaff_entropy + aff_entropy + eps)

        records.append(record)

    df = pd.DataFrame(records)

    if save_dir is not None:
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        csv_path = os.path.join(save_dir, 'ACLD_affected_vs_unaffected_attention_stats.csv')
        df.to_csv(csv_path, index=False)

    return df