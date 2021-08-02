from typing import Dict, List, Optional

from torch import Tensor
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, LightningModule

from swafa.custom_types import POSTERIOR_TYPE


class ModelPosterior:

    def __init__(self, model: LightningModule, weight_posterior_class: POSTERIOR_TYPE,
                 weight_posterior_kwargs: Optional[dict] = None):
        self.model = model
        self.weight_posterior = self._create_weight_posterior(weight_posterior_class, weight_posterior_kwargs)

    def _create_weight_posterior(self, posterior_class: POSTERIOR_TYPE, posterior_kwargs: Optional[dict] = None):
        posterior_kwargs = posterior_kwargs or dict()
        return posterior_class(self._get_weight_dimension(), **posterior_kwargs)

    def _get_weight_dimension(self) -> int:
        return sum([w.numel() for w in self.model.parameters()])

    def bayesian_model_average(self, dataloader: DataLoader, trainer: Trainer, n_samples: int) -> Tensor:
        raise NotImplementedError

    def _sample_model(self) -> LightningModule:
        raise NotImplementedError

    def test(self, dataloader: DataLoader, trainer: Trainer) -> List[Dict[str, float]]:
        raise NotImplementedError
