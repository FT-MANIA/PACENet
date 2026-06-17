import numpy as np
import torch

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import log_loss

try:
    from xgboost import XGBClassifier
    _HAS_XGBOOST = True
except Exception:
    XGBClassifier = None
    _HAS_XGBOOST = False


class FeatureBasedTraditionalML:
    is_sklearn_model = True
    def __init__(self, config):
        self.config = config
        self.model_type = config.get("model_type", "SVM")
        self.num_classes = config.get("num_classes", 3)

        self.use_raw_kf = config.get("ml_use_raw_kf", True)
        self.use_time_features = config.get("ml_use_time_features", True)

        self.classifier = self._build_classifier()
        self.classes_ = None

    def _build_classifier(self):
        if self.model_type == "SVM":
            clf = SVC(
                C=self.config.get("svm_C", 0.1),
                kernel=self.config.get("svm_kernel", "rbf"),
                gamma=self.config.get("svm_gamma", "auto"),
                probability=True,
                random_state=self.config['seed'],
            )

        elif self.model_type == "XGBoost":
            if not _HAS_XGBOOST:
                raise ImportError(
                    "xgboost is not installed. Please install it with `pip install xgboost`."
                )

            clf = XGBClassifier(
                n_estimators=self.config.get("xgb_n_estimators", 100),
                max_depth=self.config.get("xgb_max_depth", 2),
                learning_rate=self.config.get("xgb_learning_rate", 0.05),
                subsample=self.config.get("xgb_subsample", 0.5),
                colsample_bytree=self.config.get("xgb_colsample_bytree", 0.5),
                min_child_weight=self.config.get("xgb_min_child_weight", 10),
                gamma=self.config.get("xgb_gamma", 1.0),
                reg_lambda=self.config.get("xgb_reg_lambda", 5.0),
                reg_alpha=self.config.get("xgb_reg_alpha", 1.0),
                objective="multi:softprob",
                num_class=self.config.get("num_classes", 3),
                eval_metric="mlogloss",
                random_state=self.config['seed'],
                n_jobs=self.config.get("xgb_n_jobs", -1),
            )

        else:
            raise ValueError(f"Unsupported traditional ML model_type: {self.model_type}")

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf)
        ])

        return pipeline

    @staticmethod
    def _to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _band_mask(self, freqs, band):
        return (freqs >= band[0]) & (freqs <= band[1])

    def _extract_time_features(self, x_np):
        mean = x_np.mean(axis=-1)
        std = x_np.std(axis=-1)
        min_v = x_np.min(axis=-1)
        max_v = x_np.max(axis=-1)
        ptp = max_v - min_v

        base = np.concatenate([
            mean,
            std,
            min_v,
            max_v,
            ptp
        ], axis=1)

        features = [base]
        return np.concatenate(features, axis=1)

    def extract_features(self, x, raw_kf=None):
        x_np = self._to_numpy(x).astype(np.float32)

        feature_list = []

        if self.use_raw_kf and raw_kf is not None:
            raw_kf_np = self._to_numpy(raw_kf).astype(np.float32)
            if raw_kf_np.ndim == 1:
                raw_kf_np = raw_kf_np.reshape(1, -1)
            feature_list.append(raw_kf_np)

        if self.use_time_features:
            time_feat = self._extract_time_features(x_np)
            feature_list.append(time_feat.astype(np.float32))

        X_feat = np.concatenate(feature_list, axis=1)
        X_feat = np.nan_to_num(X_feat, nan=0.0, posinf=1e6, neginf=-1e6)

        return X_feat.astype(np.float32)

    def _collect_loader_features(self, loader):
        X_list = []
        y_list = []
        demo_list = []

        for batch in loader:
            x, targets, demo, raw_kf, trace_info, indices = batch

            X_feat = self.extract_features(x, raw_kf)
            y_np = self._to_numpy(targets).astype(np.int64)
            demo_np = self._to_numpy(demo).astype(np.float32)

            X_list.append(X_feat)
            y_list.append(y_np)
            demo_list.append(demo_np)

        X_all = np.concatenate(X_list, axis=0)
        y_all = np.concatenate(y_list, axis=0)
        demo_all = np.concatenate(demo_list, axis=0)

        return X_all, y_all, demo_all

    def fit_from_loader(self, train_loader):
        X_train, y_train, _ = self._collect_loader_features(train_loader)

        if self.model_type == "XGBoostFeature":
            sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
            self.classifier.fit(X_train, y_train, clf__sample_weight=sample_weight)
        else:
            self.classifier.fit(X_train, y_train)

        clf = self.classifier.named_steps["clf"]
        self.classes_ = getattr(clf, "classes_", np.arange(self.num_classes))

        return self

    def predict_prob_from_loader(self, loader):
        X, y_true, demo = self._collect_loader_features(loader)

        prob = self.classifier.predict_proba(X)
        aligned_prob = np.zeros((prob.shape[0], self.num_classes), dtype=np.float32)

        for col_idx, cls in enumerate(self.classes_):
            cls = int(cls)
            if 0 <= cls < self.num_classes:
                aligned_prob[:, cls] = prob[:, col_idx]

        row_sum = aligned_prob.sum(axis=1, keepdims=True)
        aligned_prob = aligned_prob / np.maximum(row_sum, 1e-8)

        y_pred = np.argmax(aligned_prob, axis=1)

        try:
            loss = log_loss(
                y_true,
                aligned_prob,
                labels=list(range(self.num_classes))
            )
        except Exception:
            loss = 0.0

        return y_true, y_pred, aligned_prob, demo, float(loss)