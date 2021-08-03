from typing import Dict, List, Optional

from torch import Tensor
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, LightningModule

from swafa.custom_types import POSTERIOR_TYPE


class ModelPosterior:
    """
    This class represents a model together with a posterior distribution over the model's weights.

    Args:
        model: A neural network implemented as a PyTorch Lightning model.
        weight_posterior_class: The uninitialised class which will be used to construct the posterior distribution over
            the model's weights.
        weight_posterior_kwargs: Keyword arguments which will be used when initialising the posterior class. This is
            optional, but should contain any positional arguments of the class, except the dimension of the
            distribution, which is inferred from the number of weights in the model.

    Attributes:
        weight_posterior: The initialised posterior distribution of the model's weights. Of type weight_posterior_class.
    """

    def __init__(self, model: LightningModule, weight_posterior_class: POSTERIOR_TYPE,
                 weight_posterior_kwargs: Optional[dict] = None):
        self.model = model
        self.weight_posterior = self._init_weight_posterior(weight_posterior_class, weight_posterior_kwargs)

    def _init_weight_posterior(self, posterior_class: POSTERIOR_TYPE, posterior_kwargs: Optional[dict] = None):
        """
        Initialise the posterior distribution over the parameters of the model.

        Args:
            posterior_class: The uninitialised class which will be used to construct the posterior distribution.
            posterior_kwargs: Keyword arguments which will be used when initialising the posterior class. This is
            optional, but should contain any positional arguments of the class, except the dimension of the
            distribution, which is inferred from the number of weights in the model.

        Returns:
            The initialised posterior distribution.
        """
        posterior_kwargs = posterior_kwargs or dict()
        return posterior_class(self._get_weight_dimension(), **posterior_kwargs)

    def _get_weight_dimension(self) -> int:
        """
        Get the total combined dimension of all the weights in the model.

        Returns:
            The total dimension of the model's weights.
        """
        return sum([w.numel() for w in self.model.parameters()])

    def bayesian_model_average(self, dataloader: DataLoader, trainer: Trainer, n_samples: int) -> Tensor:
        raise NotImplementedError

    def _sample_model(self) -> LightningModule:
        raise NotImplementedError

    def test(self, dataloader: DataLoader, trainer: Trainer) -> List[Dict[str, float]]:
        raise NotImplementedError
