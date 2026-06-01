import torch.nn as nn
from Models.PACENet import PACENet
from Models.ResNet import ResNet
from Models.TimesNet import TimesNet
from Models.PatchTST import PatchTST
from Models.iTransformer import iTransformer
from Models.Medformer import Medformer
from Models.ML_Models import FeatureBasedTraditionalML
import logging

def choose_model(model_type, config):
    models_map = {
        'PACENet': PACENet,
        'SVM': FeatureBasedTraditionalML,
        'XGBoost': FeatureBasedTraditionalML,
        'ResNet': ResNet,
        'TimesNet': TimesNet,
        'PatchTST': PatchTST,
        'iTransformer': iTransformer,
        'Medformer': Medformer
    }
    return models_map[model_type](config)

logger = logging.getLogger(__name__)
class exp_model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model_type = config['model_type']
        self.model = choose_model(self.model_type, config)
        self.is_sklearn_model = getattr(self.model, "is_sklearn_model", False)
        logger.info(f"Initialized Model: {self.model_type}")

    def forward(self, x, raw_kf):
        if self.model_type == 'PACENet':
            logits = self.model(x, raw_kf)
        else:
            logits = self.model(x)
        return logits

    def fit_from_loader(self, train_loader):
        return self.model.fit_from_loader(train_loader)
    def predict_prob_from_loader(self, loader):
        return self.model.predict_prob_from_loader(loader)