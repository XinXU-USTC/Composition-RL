from dataclasses import dataclass, field
from typing import Optional, Dict

Message = dict[str, str]  # keys role, content
MessageList = list[Message]


class SamplerBase:
    """
    Base class for defining a sampling model, which can be evaluated,
    or used as part of the grading process.
    """

    def __call__(self, message_list: MessageList) -> str:
        raise NotImplementedError


@dataclass
class EvalResult:
    """
    Result of running an evaluation (usually consisting of many samples)
    """

    score: Optional[float] = None   # top-line metric
    metrics: Optional[Dict[str, float]] = None  # other metrics


@dataclass
class SingleEvalResult:
    """
    Result of evaluating a single sample
    """

    score: Optional[float] = None  # top-line metric
    metrics: Dict[str, float] = field(default_factory=dict)  # other metrics with default empty dict


class Eval:
    """
    Base class for defining an evaluation.
    """

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        raise NotImplementedError
