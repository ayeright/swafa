from torch.optim import Adam, SGD


OPTIMISER_FACTORY = dict(
    sgd=SGD,
    adam=Adam,
)
