"""Module for evaluating a disentanglement model on speaker and content tasks.

Copyright (c) Jack Cox
SPDX-License-Identifier: MIT
"""

import logging
import os
from pathlib import Path

import lightning as L  # noqa: N812 standard import for lightning
import torch
from hydra.utils import instantiate
from lightning.pytorch.utilities.model_summary.model_summary import ModelSummary
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.dataset import (
    PrecomputedFeaturesDataloader,
    SpeechCommandsDataModule,
    VoxCelebPrecomputedFeaturesDataset,
    VoxCelebPrecomputedFeaturesMemMappedDataset,
    load_voxceleb_gender,
    load_voxceleb_sv_pairs,
)
from src.evaluation.embeddings import PreExtractedEmbeddingsDataModule
from src.evaluation.evaluate import evaluate_speaker
from src.evaluation.model import KeywordSpottingClassifier
from src.shared.utils import setup_loggers, visualise_latent_space
from src.train.experiment import (
    InterventionalDisentanglementExperiment,
    OptimizerFactory,
)
from src.train.model import DisentanglementModel

log = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _extract_latent_representations(
    model: DisentanglementModel,
    dataloader: DataLoader,
    stage: str = "train",
) -> tuple[Tensor, Tensor]:
    embeddings_list = []
    labels_list = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(
            dataloader,
            desc=f"Extracting {stage} embeddings",
        ):
            features, lengths, labels = batch
            features = features.to(device)
            embeddings = model.encode(features, lengths=lengths)
            embeddings_list.append(embeddings.cpu())
            labels_list.append(labels)
    all_embeddings = torch.cat(embeddings_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)
    return all_embeddings, all_labels


def _load_from_checkpoint(
    model: DisentanglementModel,
    optimizer_factory: OptimizerFactory,
    checkpoint_path: Path,
) -> InterventionalDisentanglementExperiment:
    if checkpoint_path.exists():
        log.info("Loading checkpointed disentanglement model from %s", checkpoint_path)
        experiment = InterventionalDisentanglementExperiment.load_from_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer_factory,
            map_location=device,
        )
    else:
        log.warning(
            "Checkpoint not found at %s. Evaluating with randomly "
            "initialized disentanglement model.",
            checkpoint_path,
        )
        experiment = InterventionalDisentanglementExperiment(
            model=model,
            optimizer=optimizer_factory,
        )
    return experiment


def evaluate_model(cfg: DictConfig) -> None:
    """Evaluate a disentanglement model on speaker and content tasks.

    Evaluation for a disentanglement model is performed on two tasks: speaker
    verification on VoxCeleb1 and keyword spotting classification on Speech Commands.

    Args:
        cfg (DictConfig): Configuration object containing all settings for evaluation.

    """
    log.info("Working directory: %s", Path.cwd())
    log.info("CPU count: %d", os.cpu_count())
    log.info("Thread count: %d", torch.get_num_threads())
    log.info(OmegaConf.to_yaml(cfg))  # Log config

    # Logging setup
    output_dir = Path(cfg.output_dir) / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    loggers = setup_loggers(
        cfg.job_id,
        cfg.job_name,
        cfg.logging,
        output_dir,
        job_config=cfg,
    )

    # Checkpoint and feature paths
    checkpoint_path = Path(cfg.checkpoint_dir) / "last.ckpt"

    # Seeding
    L.seed_everything(cfg.seed, workers=True)

    # ---- DATA SETUP ----
    # Load VoxCeleb1 pre-computed features dataset
    voxceleb_class = (
        VoxCelebPrecomputedFeaturesMemMappedDataset
        if cfg.eval.speaker.mmap
        else VoxCelebPrecomputedFeaturesDataset
    )
    voxceleb_dataset = voxceleb_class(
        feature_dir=cfg.eval.speaker.feature_dir,
        layers=cfg.dataset.layers,
    )
    voxceleb_dataloader = PrecomputedFeaturesDataloader(
        voxceleb_dataset,
        batch_size=cfg.eval.speaker.batch_size,
        shuffle=False,
        use_nested_tensors=cfg.eval.use_nested_tensors,
        num_workers=cfg.num_workers,
    )
    voxceleb_gender_dict = load_voxceleb_gender(cfg.eval.speaker.gender_metadata_file)

    # Load Speech Commands precomputed features
    sc_datamodule = SpeechCommandsDataModule(
        feature_dir=cfg.eval.content.feature_dir,
        layers=cfg.dataset.layers,
        batch_size=cfg.eval.content.batch_size,
        use_nested_tensors=cfg.eval.use_nested_tensors,
        num_workers=cfg.num_workers,
        mmap=cfg.eval.content.mmap,
    )
    sc_datamodule.setup(stage="fit")

    # ---- MODEL SETUP ----
    disentanglement_model = instantiate(cfg.model)

    # Load from checkpoint if available
    log.info("Loading checkpointed disentanglement model from %s", checkpoint_path)
    optimizer_factory = instantiate(cfg.optimizer)
    experiment = _load_from_checkpoint(
        model=disentanglement_model,
        optimizer_factory=optimizer_factory,
        checkpoint_path=checkpoint_path,
    )
    disentanglement_model = experiment.model
    subspaces = disentanglement_model.total_subspaces
    # Print model summary
    summary = ModelSummary(disentanglement_model, max_depth=-1)
    log.info("Disentanglement Model Summary:\n%s", summary)

    # ------ SPEAKER EVALUATION ------
    log.info("Evaluating model on VoxCeleb1 speaker verification task")
    log.info("Extracting latent representations for VoxCeleb1 test set")
    all_z = {}
    for batch in tqdm(voxceleb_dataloader):
        features, _, labels = batch
        features = features.to(device)
        with torch.no_grad():
            z = disentanglement_model.encode(features)
        for i in range(features.size(0)):
            sample_id = labels[i]
            all_z[sample_id] = z[i].cpu()  # Store latent representation for each sample

    # Plot t-SNE for each subspace
    for j in range(subspaces):
        z_j = torch.stack(
            [all_z[sample_id][j, :] for sample_id in sorted(all_z.keys())],
            dim=0,
        )
        speaker_ids = [
            int(sample_id.split("/")[0].replace("id", ""))
            for sample_id in sorted(all_z.keys())
        ]
        speaker_ids = torch.tensor(speaker_ids)
        fig = visualise_latent_space(z_j, labels=speaker_ids)
        for logger in loggers:
            if hasattr(logger, "log_image"):
                logger.log_image(key=f"eval/subspace {j}/VoxCeleb1/t-SNE", images=[fig])
        plot_path = output_dir / f"voxceleb1_tsne_subspace_{j}.png"
        fig.savefig(plot_path)

    # Load speaker verification pairs
    sv_pairs = load_voxceleb_sv_pairs(cfg.eval.speaker.sv_metadata_file)
    log.info("Loaded %d speaker verification pairs", len(sv_pairs))
    # Evaluate speaker verification
    evaluate_speaker(
        z=all_z,
        sv_pairs=sv_pairs,
        subspaces=subspaces,
        scores_path=output_dir / "voxceleb1_sv_scores.txt",
        plot_path=output_dir / "voxceleb1_sv_scores_violin.png",
        eer_path=output_dir / "voxceleb1_sv_eer.csv",
        metadata=voxceleb_gender_dict,
        logger=loggers[-1],
    )

    # ----- CONTENT EVALUATION ON SPEECH COMMANDS ------
    log.info("Evaluating model on Speech Commands keyword spotting classification task")
    log.info("Extracting latent representations for Speech Commands")

    # Extract embeddings once for all subspaces to avoid re-extraction during training
    all_train_embeddings, all_train_labels = _extract_latent_representations(
        disentanglement_model,
        sc_datamodule.train_dataloader(),
    )
    all_val_embeddings, all_val_labels = _extract_latent_representations(
        disentanglement_model,
        sc_datamodule.val_dataloader(),
    )
    all_test_embeddings, all_test_labels = _extract_latent_representations(
        disentanglement_model,
        sc_datamodule.test_dataloader(),
    )

    # Get embedding dimension
    embedding_dim = all_train_embeddings.size(-1)

    # Train classifier for each subspace
    for i in range(subspaces):
        log.info("Training keyword spotting classifier on subspace %d", i)

        # Extract embeddings for this subspace only
        train_emb = all_train_embeddings[:, i, :]  # (num_train, embedding_dim)
        val_emb = all_val_embeddings[:, i, :]
        test_emb = all_test_embeddings[:, i, :]

        # Create datamodule with pre-extracted embeddings
        emb_datamodule = PreExtractedEmbeddingsDataModule(
            train_embeddings=train_emb,
            train_labels=all_train_labels,
            val_embeddings=val_emb,
            val_labels=all_val_labels,
            test_embeddings=test_emb,
            test_labels=all_test_labels,
            batch_size=cfg.eval.content.batch_size,
            num_workers=cfg.num_workers,
        )
        emb_datamodule.setup(stage="fit")

        classifier = KeywordSpottingClassifier(
            embedding_dim=embedding_dim,
            output_class_num=sc_datamodule.num_classes,
            class_mappings=sc_datamodule.class_mappings,
            subspace=i,
            save_dir=output_dir,
        )
        trainer = L.Trainer(
            logger=loggers,
            **cfg.eval.content.trainer,
            inference_mode=False,
        )
        trainer.fit(classifier, datamodule=emb_datamodule)
        trainer.test(classifier, datamodule=emb_datamodule)


if __name__ == "__main__":
    import hydra

    from src.shared.utils import hydra_dirname

    if not OmegaConf.has_resolver("dirname"):
        OmegaConf.register_new_resolver("dirname", resolver=hydra_dirname)

    @hydra.main(version_base=None, config_path="pkg://configs", config_name="default")
    def main(cfg: DictConfig) -> None:
        """Hydra entry point for evaluation.

        Args:
            cfg (DictConfig): Configuration object containing all settings for
            evaluation.

        """
        try:
            evaluate_model(cfg)
        except Exception:
            log.exception("An error occurred during evaluation.")
            raise

    main()
