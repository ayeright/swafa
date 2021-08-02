from typing import Union

from swafa.fa import OnlineGradientFactorAnalysis, OnlineEMFactorAnalysis


POSTERIOR_TYPE = Union[OnlineGradientFactorAnalysis, OnlineEMFactorAnalysis]
