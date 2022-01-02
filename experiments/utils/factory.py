from torch.nn import ReLU
from torch.optim import Adam, SGD


OPTIMISER_FACTORY = dict(
    sgd=SGD,
    adam=Adam,
)

ACTIVATION_FACTORY = dict(
    relu=ReLU(),
)
