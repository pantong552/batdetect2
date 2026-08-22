"""Scheduler configuration and factory utilities for training."""

from typing import Annotated, Literal

from loguru import logger
from pydantic import Field
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler

from batdetect2.core import (
    BaseConfig,
    ImportConfig,
    Registry,
    add_import_config,
)

__all__ = [
    "CosineAnnealingSchedulerConfig",
    "SchedulerConfig",
    "SchedulerImportConfig",
    "build_scheduler",
    "scheduler_registry",
]


scheduler_registry: Registry[LRScheduler, [Optimizer]] = Registry("scheduler")


@add_import_config(scheduler_registry, arg_names=["optimizer"])
class SchedulerImportConfig(ImportConfig):
    """Use any callable as a scheduler.

    Set ``name="import"`` and provide a ``target`` pointing to any callable
    that returns a scheduler. The optimizer instance is passed as the
    ``optimizer`` keyword argument.
    """

    name: Literal["import"] = "import"


class CosineAnnealingSchedulerConfig(BaseConfig):
    """Configuration for ``CosineAnnealingLR`` with optional Linear Warmup.

    Attributes
    ----------
    name : Literal["cosine_annealing"]
        Discriminator field used by the scheduler registry.
    t_max : int
        Number of epochs to complete one cosine cycle.
    eta_min : float, optional
        Minimum learning rate. Defaults to 0.
    warmup_epochs : int, optional
        Number of linear warmup epochs from start_factor to 1.0. Defaults to 0.
    warmup_start_factor : float, optional
        Initial learning rate multiplier during warmup. Defaults to 0.1.
    """

    name: Literal["cosine_annealing"] = "cosine_annealing"
    t_max: int = 200
    eta_min: float = 0
    warmup_epochs: int = 0
    warmup_start_factor: float = 0.1


@scheduler_registry.register(CosineAnnealingSchedulerConfig)
def build_cosine_scheduler(
    config: CosineAnnealingSchedulerConfig,
    optimizer: Optimizer,
) -> LRScheduler:
    """Build a cosine annealing scheduler with optional linear warmup.

    ``t_max`` is interpreted in epochs because Lightning steps the scheduler
    once per epoch when ``interval="epoch"`` is used.
    """
    if config.warmup_epochs > 0:
        from torch.optim.lr_scheduler import LinearLR, SequentialLR

        warmup_sched = LinearLR(
            optimizer,
            start_factor=config.warmup_start_factor,
            end_factor=1.0,
            total_iters=config.warmup_epochs,
        )
        cosine_t_max = max(1, config.t_max - config.warmup_epochs)
        cosine_sched = CosineAnnealingLR(
            optimizer,
            T_max=cosine_t_max,
            eta_min=config.eta_min,
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[config.warmup_epochs],
        )

    return CosineAnnealingLR(
        optimizer,
        T_max=config.t_max,
        eta_min=config.eta_min,
    )


SchedulerConfig = Annotated[
    CosineAnnealingSchedulerConfig | SchedulerImportConfig,
    Field(discriminator="name"),
]


def build_scheduler(
    optimizer: Optimizer,
    config: SchedulerConfig | None = None,
) -> LRScheduler:
    """Build a scheduler from configuration."""
    config = config or CosineAnnealingSchedulerConfig()

    logger.opt(lazy=True).debug(
        "Building scheduler with config: \n{}",
        lambda: config.to_yaml_string(),
    )

    return scheduler_registry.build(config, optimizer)
