import logging
import argparse
import os
import pandas as pd
import pickle

from utils import setup, device_init, setup_seed
from Dataset.dataset_loader import load_gait_data
from trainer import kfold_runner
from plot_utils import (plot_kfold_confusion_matrix, plot_class_spectral_energy, plot_multi_model_roc_comparison,
                        plot_multi_model_metric_boxplots, plot_ablation_metric_boxplots, plot_paired_delta_metric,
                        subset_extractor, attn_weight_extractor, plot_raw_signal_attention, plot_spectral_weights,
                        subset_analysis)

logger = logging.getLogger('__main__')
parser = argparse.ArgumentParser()

parser.add_argument('--run_mode', type=str, default='replot', choices=['main', 'exp', 'replot'],
                    help="running mode: 'main': only for PACENet, 'exp': for contrast or ablation exp, 'replot': for ploting pic paper need")
parser.add_argument('--dataset_path', default='Dataset/KGKD', help='Data path for dataset')
parser.add_argument('--output_dir', default='Results', help='Output directory')
parser.add_argument('--gpu', type=str, default='0', help='GPU index')
parser.add_argument('--seed', default=172, type=int, help='Random seed')

parser.add_argument('--internal_test_size', default=0.2, type=float, help='Size of internal test set of development set')
parser.add_argument('--Norm', type=bool, default=True, help='Enable Data Normalization')
parser.add_argument("--use_data_augmentation", type=bool, default=True, help="Enable data augmentation")
parser.add_argument("--aug_ratios", type=dict, default={0: 2, 1: 2, 2: 4}, help="Ratios of augmentations per class")

parser.add_argument("--kernel_size", default=[6, 40], type=int, nargs=2, help="kernel size of UFE")
parser.add_argument("--stride", default=3, type=int, help="stride size of UFE")
parser.add_argument("--num_heads", default=4, type=int, help="Number of attention heads")
parser.add_argument("--embed_dim", default=64, type=int, help="Dimension of embedding")
parser.add_argument("--dim_ff", default=256, type=int, help="Dimension of feedforward layer")
parser.add_argument("--UFE_dim", default=32, type=int, help="Dimension of UFE output")
parser.add_argument("--CFE_dim", default=32, type=int, help="Dimension of CFE output")
parser.add_argument("--SFE_dim", default=32, type=int, help="Dimension of SFE output")
parser.add_argument("--num_KF", default=12, type=int, help="Number of kinematic features")
parser.add_argument("--KFE_dim", default=32, type=int, help="Dimension of KFE output")
parser.add_argument("--dropout", default=0.4, type=float, help="Dropout rate")
parser.add_argument("--num_classes", default=3, type=int, help="Number of classes for final output")

parser.add_argument("--batch_size", default=64, type=int, help="Batch size")
parser.add_argument("--lr", default=3e-4, type=float, help="Learning rate (Base)")
parser.add_argument("--weight_decay", default=1e-2, type=float, help="Weight decay")
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--k_folds", default=5, type=int, help="whether use cycle voting")

parser.add_argument("--model_type", default='PACENet', type=str, help="model type for knee")
parser.add_argument("--use_UFE", default=True, type=bool, help="whether use UFE")
parser.add_argument("--use_SFE", default=True, type=bool, help="whether use SFE")
parser.add_argument("--use_KFE", default=True, type=bool, help="whether use KFE")
parser.add_argument("--use_CFE", default=True, type=bool, help="whether use CFE")

def get_arg():
    return parser.parse_args()

def run_main_analysis(config, device, Data):
    logger.info("=" * 50)
    logger.info("Running mode: Main Analysis For PACENet")
    logger.info("=" * 50)

    exp_name = "Main_Analysis"
    results = kfold_runner(exp_name, {'ks': [6, 40]}, Data, config)

    dev_test_reports = results['dev_test_reports']
    ext_test_reports = results['ext_test_reports']
    class_names = ['Healthy', 'ACLD', 'KOA']

    ext_test_subset_reports, dev_test_subset_reports = subset_extractor(ext_test_reports, dev_test_reports, class_names)
    attn_weight = attn_weight_extractor(results, device, class_names)
    plot_data_package = {
        'dev_test_reports': dev_test_reports,
        'ext_test_reports': ext_test_reports,
        'dev_test_subset_reports': dev_test_subset_reports,
        'ext_test_subset_reports': ext_test_subset_reports,
        'spectral_weights': results['spectral_weights'],
        'attn_weight': attn_weight
    }

    pkl_path = os.path.join(config['plot_data_dir'], 'latest_plot_data.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(plot_data_package, f)

    logger.info(f"Exp data of main analysis have been collected and saved in {pkl_path}")
    return plot_data_package

def run_exps(config, Data):
    logger.info("=" * 50)
    logger.info("Running mode: All Exps of the paper (Benchmark & Ablation)")
    logger.info("=" * 50)

    experiments = []

    experiments.append(('Ablation: Full PACENet', {
        'model_type': 'PACENet'
    }))

    experiments.append(('Ablation: wo UFE', {
        'use_UFE': False
    }))

    experiments.append(('Ablation: wo CFE', {
        'use_CFE': False
    }))

    experiments.append(('Ablation: wo SFE', {
        'use_SFE': False
    }))

    experiments.append(('Ablation: wo KFE', {
        'use_KFE': False
    }))

    experiments.append(('Aug_Ablation: Full PACENet', {
        'model_type': 'PACENet'
    }))

    experiments.append(('Aug_Ablation: wo Data Augmentation', {
        'use_data_augmentation': False
    }))

    experiments.append(('Aug_Ablation: wo Jitter', {
        'use_jitter': False
    }))

    experiments.append(('Aug_Ablation: wo Scaling', {
        'use_scaling': False
    }))

    experiments.append(('Aug_Ablation: wo Magnitude Warp', {
        'use_magnitude_warp': False
    }))

    experiments.append(('Aug_Ablation: wo Time Warp', {
        'use_time_warp': False
    }))

    experiments.append(('Aug_Ablation: wo Random Bias', {
        'use_random_bias': False
    }))

    experiments.append(('Aug_Ablation: wo Crosstalk', {
        'use_crosstalk': False
    }))

    comparison_models = [
        'PACENet',
        'ResNet',
        'TimesNet',
        'PatchTST',
        'iTransformer',
        'Medformer',
        'SVM',
        'XGBoost'
    ]

    for model_name in comparison_models:
        experiments.append((f'{model_name}', {'model_type': model_name }))

    all_results = []
    benchmark_pkl_path = os.path.join(config['plot_data_dir'], 'benchmark_plot_data.pkl')
    ablation_pkl_path = os.path.join(config['plot_data_dir'], 'ablation_plot_data.pkl')

    if os.path.exists(benchmark_pkl_path):
        with open(benchmark_pkl_path, 'rb') as f1:
            benchmark_plot_data = pickle.load(f1)
    else:
        benchmark_plot_data = {}
    if os.path.exists(ablation_pkl_path):
        with open(ablation_pkl_path, 'rb') as f2:
            ablation_pkl_data = pickle.load(f2)
    else:
        ablation_pkl_data = {}

    logger.info(f"Plan to run {len(experiments)} experiments...")
    for exp_name, exp_config in experiments:
        setup_seed(config)
        results = kfold_runner(exp_name, exp_config, Data, config)
        all_results.append(results['summary'])

        if "Ablation" not in exp_name:
            benchmark_plot_data[exp_name] = {
                'dev_test_y_true_list': [r['y_true'] for r in results['dev_test_reports']],
                'dev_test_y_prob_list': [r['y_prob'] for r in results['dev_test_reports']],
                'ext_test_y_true_list': [r['y_true'] for r in results['ext_test_reports']],
                'ext_test_y_prob_list': [r['y_prob'] for r in results['ext_test_reports']],
            }

        if "Ablation" in exp_name:
            exp_name = exp_name.replace("Ablation: ", "")
            exp_name = exp_name.replace("wo", "w/o")
            ablation_pkl_data[exp_name] = {
                'dev_test_y_true_list': [r['y_true'] for r in results['dev_test_reports']],
                'dev_test_y_prob_list': [r['y_prob'] for r in results['dev_test_reports']],
                'ext_test_y_true_list': [r['y_true'] for r in results['ext_test_reports']],
                'ext_test_y_prob_list': [r['y_prob'] for r in results['ext_test_reports']],
            }

    with open(benchmark_pkl_path, 'wb') as f1:
        pickle.dump(benchmark_plot_data, f1)
    with open(ablation_pkl_path, 'wb') as f2:
        pickle.dump(ablation_pkl_data, f2)

    final_df = pd.DataFrame(all_results)
    final_path = os.path.join(config['save_dir'], 'Exp_Report.csv')
    final_df.to_csv(final_path, index=False)
    logger.info(f"All Done! Report saved to: {final_path}")
    seed = None

    return final_df, final_path, seed

def render_all_plots(config, Data):
    plot_class_spectral_energy(
        x=Data['dev_data'],
        y=Data['dev_label'],
        class_names=['Healthy', 'ACLD', 'KOA'],
        fs=60,
        save_dir=config['plot_data_dir'],
        name='dev_dataset_class_spectral.png'
    )
    plot_class_spectral_energy(
        x=Data['dev_test_data'],
        y=Data['dev_test_label'],
        class_names=['Healthy', 'ACLD', 'KOA'],
        fs=60,
        save_dir=config['plot_data_dir'],
        name='dev_test_dataset_class_spectral.png'
    )
    plot_class_spectral_energy(
        x=Data['ext_test_data'],
        y=Data['ext_test_label'],
        class_names=['Healthy', 'ACLD', 'KOA'],
        fs=60,
        save_dir=config['plot_data_dir'],
        name='ext_test_dataset_class_spectral.png'
    )

    pkl_path = os.path.join(config['plot_data_dir'], 'latest_plot_data.pkl')
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            plot_data = pickle.load(f)
        dev_test_reports = plot_data['dev_test_reports']
        ext_test_reports = plot_data['ext_test_reports']
        class_names = ['Healthy', 'ACLD', 'KOA']

        plot_kfold_confusion_matrix([r['confusion_matrix'] for r in dev_test_reports], class_names, config['plot_data_dir'],
                                    'kfold_overall_dev_test_CM.png')
        plot_kfold_confusion_matrix([r['confusion_matrix'] for r in ext_test_reports], class_names, config['plot_data_dir'],
                                    'kfold_overall_ext_test_CM.png')

        subset_analysis(plot_data['dev_test_subset_reports'], config['plot_data_dir'], prefix='Dev_Test')
        subset_analysis(plot_data['ext_test_subset_reports'], config['plot_data_dir'], prefix='Ext_Test')

        plot_raw_signal_attention(plot_data, config['plot_data_dir'])
        plot_spectral_weights(plot_data, config['plot_data_dir'])

    pkl_path = os.path.join(config['plot_data_dir'], 'benchmark_plot_data.pkl')
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            plot_data = pickle.load(f)
        plot_multi_model_roc_comparison(plot_data, config['num_classes'], config['plot_data_dir'], model='dev_test',
                                        name='multi_model_roc_comparison_dev_test.png')
        plot_multi_model_roc_comparison(plot_data, config['num_classes'], config['plot_data_dir'], model='ext_test',
                                        name='multi_model_roc_comparison_ext_test.png')

        plot_multi_model_metric_boxplots(plot_data, config['num_classes'], config['plot_data_dir'], model='dev_test',
                                         metric='f1')
        plot_multi_model_metric_boxplots(plot_data, config['num_classes'], config['plot_data_dir'], model='ext_test',
                                         metric='f1')
        plot_multi_model_metric_boxplots(plot_data, config['num_classes'], config['plot_data_dir'], model='dev_test',
                                         metric='accuracy')
        plot_multi_model_metric_boxplots(plot_data, config['num_classes'], config['plot_data_dir'], model='ext_test',
                                         metric='accuracy')

    pkl_path = os.path.join(config['plot_data_dir'], 'ablation_plot_data.pkl')
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            plot_data = pickle.load(f)
        plot_ablation_metric_boxplots(plot_data, config['num_classes'], config['plot_data_dir'], model='dev_test',
                                      metric='f1', experiment_group='architecture')
        plot_ablation_metric_boxplots(plot_data, config['num_classes'], config['plot_data_dir'], model='ext_test',
                                      metric='f1', experiment_group='architecture')
        plot_ablation_metric_boxplots(plot_data, config['num_classes'], config['plot_data_dir'], model='dev_test',
                                      metric='accuracy', experiment_group='architecture')
        plot_ablation_metric_boxplots(plot_data, config['num_classes'], config['plot_data_dir'], model='ext_test',
                                      metric='accuracy', experiment_group='architecture')

        plot_ablation_metric_boxplots(plot_data, config['num_classes'], config['plot_data_dir'], model='dev_test',
                                      metric='f1', experiment_group='augmentation')
        plot_ablation_metric_boxplots(plot_data, config['num_classes'], config['plot_data_dir'], model='ext_test',
                                      metric='f1', experiment_group='augmentation')
        plot_ablation_metric_boxplots(plot_data, config['num_classes'], config['plot_data_dir'], model='dev_test',
                                      metric='accuracy', experiment_group='augmentation')
        plot_ablation_metric_boxplots(plot_data, config['num_classes'], config['plot_data_dir'], model='ext_test',
                                      metric='accuracy', experiment_group='augmentation')

        plot_paired_delta_metric(plot_data, config['num_classes'], config['plot_data_dir'], model='dev_test',
                                 metric='f1', experiment_group='architecture')
        plot_paired_delta_metric(plot_data, config['num_classes'], config['plot_data_dir'], model='ext_test',
                                 metric='f1', experiment_group='architecture')
        plot_paired_delta_metric(plot_data, config['num_classes'], config['plot_data_dir'], model='dev_test',
                                 metric='accuracy', experiment_group='architecture')
        plot_paired_delta_metric(plot_data, config['num_classes'], config['plot_data_dir'], model='ext_test',
                                 metric='accuracy', experiment_group='architecture')

        plot_paired_delta_metric(plot_data, config['num_classes'], config['plot_data_dir'], model='dev_test',
                                 metric='f1', experiment_group='augmentation')
        plot_paired_delta_metric(plot_data, config['num_classes'], config['plot_data_dir'], model='ext_test',
                                 metric='f1', experiment_group='augmentation')
        plot_paired_delta_metric(plot_data, config['num_classes'], config['plot_data_dir'], model='dev_test',
                                 metric='accuracy', experiment_group='augmentation')
        plot_paired_delta_metric(plot_data, config['num_classes'], config['plot_data_dir'], model='ext_test',
                                 metric='accuracy', experiment_group='augmentation')

if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    config = setup(get_arg())
    setup_seed(config)
    device = device_init(config)
    config['device'] = device
    mode = config['run_mode']

    if mode == 'main':
        Data = load_gait_data(config)
        run_main_analysis(config, device, Data)

    elif mode == 'exp':
        Data = load_gait_data(config)
        run_exps(config, Data)

    elif mode == 'replot':
        Data = load_gait_data(config)
        render_all_plots(config, Data)

