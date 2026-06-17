"""Module with main entry point for sequential training and evaluation.

Copyright (c) Jack Cox
SPDX-License-Identifier: MIT
"""

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from src.evaluate_model import evaluate_model
from src.shared.utils import hydra_dirname
from src.train_model import train_model

log = logging.getLogger(__name__)
if not OmegaConf.has_resolver("dirname"):
    OmegaConf.register_new_resolver("dirname", resolver=hydra_dirname)


@hydra.main(version_base=None, config_path="pkg://configs", config_name="default")
def main(cfg: DictConfig) -> None:
    """Train and evaluate a model.

    Args:
        cfg (DictConfig): Configuration object containing all settings for training and
        evaluation.

    """
    try:
        train_model(cfg)
        evaluate_model(cfg)
    except Exception:
        log.exception("An error occurred during training or evaluation.")
        raise


if __name__ == "__main__":
    main()
