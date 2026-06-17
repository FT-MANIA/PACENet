import torch
import numpy as np
import logging
import os
import pandas as pd
from utils import compute_metrics
from tqdm import tqdm
from collections import defaultdict
from Dataset.dataset_loader import build_dataset
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from Dataset.data_augmentation import data_augmentor
from Models.Exp_Models import exp_model
from clinical_screening import (
    fit_threshold_by_policy,
    evaluate_screening_at_threshold,
    evaluate_thresholded_three_class,
    add_prefix,
    summarize_screening_records,
)

logger = logging.getLogger('__main__')
class Trainer:
    def __init__(self, model, train_loader, val_loader, dev_test_loader, ext_test_loader, loss_function, optimizer, config=None):
        self.model = model
        self.loaders = {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'dev_test_loader': dev_test_loader,
            'ext_test_loader': ext_test_loader
        }
        self.device = config['device']
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.config = config

    def train_epoch(self, epoch):
        self.model.train()
        optimizer = self.optimizer
        loss_function = self.loss_function
        train_loader = self.loaders['train_loader']

        epoch_loss = 0
        total_samples = 0

        for batch in train_loader:
            x, targets, demo, raw_kf, trace_info, indices = batch
            x, targets, demo, raw_kf = x.to(self.device), targets.to(self.device), demo.to(self.device), raw_kf.to(self.device)

            logits = self.model(x, raw_kf)
            loss = loss_function(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss = loss.item() * len(targets)
            epoch_loss += batch_loss
            total_samples += len(targets)

        return {'epoch': epoch, 'loss': epoch_loss / total_samples}

    def eval_epoch(self, flag, epoch):
        self.model.eval()
        loader = self.loaders[f'{flag}_loader']
        loss_function = self.loss_function

        all_preds, all_targets = [], []
        all_probs = []
        all_demos = []
        epoch_loss = 0
        total_samples = 0

        with torch.no_grad():
            for batch in loader:
                x, targets, demo, raw_kf, trace_info, indices = batch
                x, targets, demo, raw_kf = x.to(self.device), targets.to(self.device), demo.to(self.device), raw_kf.to(self.device)

                logits = self.model(x, raw_kf)
                loss = loss_function(logits, targets)
                batch_loss = loss.item() * len(targets)
                epoch_loss += batch_loss
                total_samples += len(targets)

                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)

                all_probs.append(probs.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                all_demos.append(demo.cpu().numpy())

        avg_loss = epoch_loss / total_samples if total_samples > 0 else 0
        if len(all_preds) > 0 and isinstance(all_preds[0], np.ndarray):
            all_preds = np.concatenate(all_preds)
            all_targets = np.concatenate(all_targets)
            all_probs = np.concatenate(all_probs)
            all_demos = np.concatenate(all_demos)

        metrics = compute_metrics(
            all_targets, all_preds, all_probs, avg_loss, epoch,
            target_names=['Healthy', 'ACLD', 'KOA'],
        )
        metrics['y_true'] = all_targets
        metrics['y_pred'] = all_preds
        metrics['y_prob'] = all_probs
        metrics['y_demo'] = all_demos

        return metrics

def train_runner(config, model, trainer, epochs, fold, exp_name=''):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trainer.optimizer, epochs, eta_min=1e-6)
    best_f1 = 0.0
    train_loss, val_loss = [], []
    best_val_metrics = None

    pbar = tqdm(range(epochs), desc=f'Training Fold {fold + 1}', leave=False)
    for epoch in pbar:
        metrics_train = trainer.train_epoch(epoch)
        train_loss.append(metrics_train['loss'])

        metrics_val = trainer.eval_epoch(flag='val', epoch=epoch)
        val_loss.append(metrics_val['loss'])
        scheduler.step()

        if metrics_val['f1_macro'] > best_f1:
            best_f1 = metrics_val['f1_macro']
            best_val_metrics = metrics_val
            torch.save(model.state_dict(),
                       os.path.join(config['save_dir'], f'{exp_name}_fold_{fold + 1}_model_best.pth'))

        pbar.set_postfix({'Tr_Loss': f"{metrics_train['loss']:.4f}", 'Val_Acc': f"{metrics_val['accuracy']:.4f}"})

    best_model_path = os.path.join(config['save_dir'], f'{exp_name}_fold_{fold + 1}_model_best.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
    metrics_dev_test = trainer.eval_epoch(flag='dev_test', epoch=0)
    metrics_ext_test = trainer.eval_epoch(flag='ext_test', epoch=0)

    return best_val_metrics, metrics_dev_test, metrics_ext_test, train_loss, val_loss

def eval_sklearn_epoch(model, loader,epoch=0):
    y_true, y_pred, y_prob, y_demo, avg_loss = model.predict_prob_from_loader(loader)

    metrics = compute_metrics(
        y_true,
        y_pred,
        y_prob,
        avg_loss,
        epoch,
        target_names=['Healthy', 'ACLD', 'KOA'],
    )

    metrics['y_true'] = y_true
    metrics['y_pred'] = y_pred
    metrics['y_prob'] = y_prob
    metrics['y_demo'] = y_demo

    return metrics

def sklearn_runner(config, model, train_loader, val_loader, dev_test_loader, ext_test_loader, fold):
    logger.info(f"[TraditionalML] Fitting {config['model_type']} on fold {fold + 1}...")
    model.fit_from_loader(train_loader)

    best_val_metrics = eval_sklearn_epoch(model, val_loader, epoch=0)
    metrics_dev_test = eval_sklearn_epoch(model, dev_test_loader, epoch=0)
    metrics_ext_test = eval_sklearn_epoch(model, ext_test_loader, epoch=0)

    train_loss = []
    val_loss = [best_val_metrics.get('loss', 0.0)]

    return best_val_metrics, metrics_dev_test, metrics_ext_test, train_loss, val_loss

def kfold_runner(exp_name, config_override, data, base_config):
    config = base_config.copy()
    config.update(config_override)
    logger.info(f"\n{'=' * 50}\nRunning Experiment: {exp_name}\nConfig Overrides: {config_override}\n{'=' * 50}")

    x_dev, y_dev, demo_dev, trace_info_dev = data['dev_data'], data['dev_label'], data['dev_demo'], data['dev_trace_info']
    x_dev_test, y_dev_test, demo_dev_test, trace_info_dev_test = data['dev_test_data'], data['dev_test_label'], data['dev_test_demo'], data['dev_test_trace_info']
    x_ext_test, y_ext_test, demo_ext_test, trace_info_test = data['ext_test_data'], data['ext_test_label'], data['ext_test_demo'], data['ext_test_trace_info']

    skf = StratifiedKFold(n_splits=config['k_folds'], shuffle=True, random_state=config['seed'])

    dev_test_fold_metrics = defaultdict(list)
    ext_test_fold_metrics = defaultdict(list)

    keys_to_track = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'auroc_macro', 'auprc_macro']

    dev_test_reports, ext_test_reports = [], []
    all_train_loss, all_val_loss = [], []
    spectral_weights = []
    clinical_screening_records = []
    best_model, ext_test_loader = None, None

    for fold, (train_idx, val_idx) in enumerate(skf.split(x_dev, y_dev)):
        logger.info(f"➜ Fold {fold + 1}/{config['k_folds']}")

        x_train_fold, y_train_fold, demo_train_fold, trace_info_train_fold = x_dev[train_idx], y_dev[train_idx], demo_dev[train_idx], trace_info_dev[train_idx]
        x_val_fold, y_val_fold, demo_val_fold, trace_info_val_fold = x_dev[val_idx], y_dev[val_idx], demo_dev[val_idx], trace_info_dev[val_idx]

        train_dataset = build_dataset(x_train_fold, y_train_fold, demo_train_fold, trace_info_train_fold, config,
                                      scaler=None, is_train=True)
        fitted_scalers = train_dataset.scalers

        if config['use_data_augmentation']:
            aug_config = {'jitter_std': 0.01, 'scale_range': [0.95, 1.05], 'rotation_range': [-5, 5],
                          'time_warp_knots': 4, 'magnitude_warp_std': 0.1, 'bias_range': [-3, 3],

                          'use_jitter': config.get('use_jitter', True),
                          'use_scaling': config.get('use_scaling', True),
                          'use_magnitude_warp': config.get('use_magnitude_warp', True),
                          'use_time_warp': config.get('use_time_warp', True),
                          'use_random_bias': config.get('use_random_bias', True),
                          'use_crosstalk': config.get('use_crosstalk', True)
                          }

            x_train_fold, y_train_fold, demo_train_fold, trace_info_train_fold = data_augmentor(x_train_fold, y_train_fold,
                                                                                demographics=demo_train_fold, trace_info=trace_info_train_fold,
                                                                                config=aug_config, aug_ratios=config['aug_ratios'])

        train_dataset = build_dataset(x_train_fold, y_train_fold, demo_train_fold, trace_info_train_fold, config,
                                      scaler=fitted_scalers, is_train=False)

        val_dataset = build_dataset(x_val_fold, y_val_fold, demo_val_fold, trace_info_val_fold, config,
                                    scaler=fitted_scalers, is_train=False)
        dev_test_dataset = build_dataset(x_dev_test, y_dev_test, demo_dev_test, trace_info_dev_test, config,
                                         scaler=fitted_scalers, is_train=False)
        ext_test_dataset = build_dataset(x_ext_test, y_ext_test, demo_ext_test, trace_info_test,config,
                                          scaler=fitted_scalers, is_train=False)

        config['ts_dim'], config['ts_len'] = train_dataset[0][0].shape[0], train_dataset[0][0].shape[1]

        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
        dev_test_loader = DataLoader(dev_test_dataset, batch_size=config['batch_size'], shuffle=False)
        ext_test_loader = DataLoader(ext_test_dataset, batch_size=config['batch_size'], shuffle=False)

        model = exp_model(config)
        if getattr(model, "is_sklearn_model", False):
            best_val_metrics, dev_test_metrics, ext_test_metrics, fold_train_loss, fold_val_loss = sklearn_runner(
                config=config,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                dev_test_loader=dev_test_loader,
                ext_test_loader=ext_test_loader,
                fold=fold
            )
        else:
            model = model.to(config['device'])
            weight = torch.tensor([1.0, 1.0, 1.0]).to(config['device'])
            loss_function = torch.nn.CrossEntropyLoss(weight=weight, label_smoothing=0.1)
            optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

            trainer = Trainer(model, train_loader, val_loader, dev_test_loader, ext_test_loader, loss_function, optimizer, config)

            best_val_metrics, dev_test_metrics, ext_test_metrics, fold_train_loss, fold_val_loss = train_runner(
                config=config,
                model=model,
                trainer=trainer,
                epochs=config['epochs'],
                fold=fold,
                exp_name=exp_name
            )

            if config.get("enable_clinical_screening", True):
                abnormal_classes = tuple(config.get("screen_abnormal_classes", [1, 2]))
                screening_policies = config.get(
                    "screening_policies",
                    [
                        ("sensitivity", 0.95),
                        ("sensitivity", 0.98),
                        ("specificity", 0.95),
                    ]
                )

                for policy, target_value in screening_policies:
                    screen_val_fit = fit_threshold_by_policy(
                        y_true=best_val_metrics["y_true"],
                        y_prob=best_val_metrics["y_prob"],
                        policy=policy,
                        target_value=target_value,
                        abnormal_classes=abnormal_classes
                    )

                    threshold = screen_val_fit["threshold"]
                    policy_name = screen_val_fit["threshold_policy"]

                    screen_val = evaluate_screening_at_threshold(
                        y_true=best_val_metrics["y_true"],
                        y_prob=best_val_metrics["y_prob"],
                        threshold=threshold,
                        abnormal_classes=abnormal_classes
                    )

                    screen_dev = evaluate_screening_at_threshold(
                        y_true=dev_test_metrics["y_true"],
                        y_prob=dev_test_metrics["y_prob"],
                        threshold=threshold,
                        abnormal_classes=abnormal_classes
                    )

                    screen_ext = evaluate_screening_at_threshold(
                        y_true=ext_test_metrics["y_true"],
                        y_prob=ext_test_metrics["y_prob"],
                        threshold=threshold,
                        abnormal_classes=abnormal_classes
                    )

                    screen_dev_3cls = evaluate_thresholded_three_class(
                        y_true=dev_test_metrics["y_true"],
                        y_prob=dev_test_metrics["y_prob"],
                        threshold=threshold
                    )

                    screen_ext_3cls = evaluate_thresholded_three_class(
                        y_true=ext_test_metrics["y_true"],
                        y_prob=ext_test_metrics["y_prob"],
                        threshold=threshold
                    )

                    record = {
                        "experiment": exp_name,
                        "fold": fold + 1,
                        "policy": policy,
                        "target_value": target_value,
                        "policy_name": policy_name,
                        "threshold": threshold,
                    }

                    record.update(add_prefix(screen_val, "val_screen"))
                    record.update(add_prefix(screen_dev, "dev_screen"))
                    record.update(add_prefix(screen_ext, "ext_screen"))
                    record.update(add_prefix(screen_dev_3cls, "dev"))
                    record.update(add_prefix(screen_ext_3cls, "ext"))

                    clinical_screening_records.append(record)

        dev_test_reports.append(dev_test_metrics)
        ext_test_reports.append(ext_test_metrics)

        all_train_loss.append(fold_train_loss)
        all_val_loss.append(fold_val_loss)

        try:
            weights = model.model.SFE.proj[1].weight.detach().cpu().numpy()
            spectral_weights.append(weights)
        except:
            pass

        best_model = model
        logger.info(f"Fold {fold + 1} Finished. Dev Acc: {dev_test_metrics['accuracy']:.4f} Test Acc: {ext_test_metrics['accuracy']:.4f}")

        for k in keys_to_track:
            dev_test_fold_metrics[k].append(dev_test_metrics.get(k, 0.0))
            ext_test_fold_metrics[k].append(ext_test_metrics.get(k, 0.0))

    summary = {'Experiment': exp_name}
    for k in keys_to_track:
        dev_test_values = dev_test_fold_metrics[k]
        ext_test_values = ext_test_fold_metrics[k]
        summary[f'dev_test_{k}_str'] = f"{np.mean(dev_test_values):.4f} ± {np.std(dev_test_values):.4f}"
        summary[f'ext_test_{k}_str'] = f"{np.mean(ext_test_values):.4f} ± {np.std(ext_test_values):.4f}"

    logger.info(f"Experiment {exp_name} Completed. Dev_Acc: {summary.get('dev_test_accuracy_str', 'N/A')} Test_Acc: {summary.get('ext_test_accuracy_str', 'N/A')}")

    if config.get("enable_clinical_screening", True) and len(clinical_screening_records) > 0:
        screening_df = pd.DataFrame(clinical_screening_records)

        safe_exp_name = (
            exp_name.replace(" ", "_")
            .replace("/", "_")
            .replace(":", "")
            .replace("\\", "_")
        )

        screening_path = os.path.join(
            config["save_dir"],
            f"{safe_exp_name}_clinical_screening_folds.csv"
        )

        screening_df.to_csv(screening_path, index=False)
        logger.info(f"[Clinical Screening] Fold-level screening metrics saved to: {screening_path}")

        for policy_name, df_policy in screening_df.groupby("policy_name"):
            records_policy = df_policy.to_dict("records")
            screening_summary = summarize_screening_records(records_policy)

            for k, v in screening_summary.items():
                summary[f"clinical_{policy_name}_{k}"] = v

    return {
        'summary': summary,
        'dev_test_reports': dev_test_reports,
        'ext_test_reports': ext_test_reports,
        'train_loss': all_train_loss,
        'val_loss': all_val_loss,
        'spectral_weights': spectral_weights,
        'best_model': best_model,
        'ext_test_loader': ext_test_loader
    }

