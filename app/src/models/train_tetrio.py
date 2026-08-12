import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import TQDMProgressBar

from maikol_utils.print_utils import print_separator
from maikol_utils.file_utils import make_dirs

from src.data import load_precomputed_tetrio_data
from src.models import TurboMinoModule, TetrisEnv
from src.config import Configuration
from src.tetris import TetrisConfiguration
from .test import test_model


def train_tetrio_turbomino(CONFIG: Configuration, T_CONFIG: TetrisConfiguration):
    """
    Train a tetrio classification model using the provided configuration.
    Args:
        CONFIG (Configuration): The configuration object containing training parameters.
        T_CONFIG (TetrisConfiguration): The configuration object containing tetris-specific parameters.
    """
    pl.seed_everything(CONFIG.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    checkpoint_dir = os.path.join(CONFIG.pretrain_model_path, CONFIG.exp_name)
    make_dirs([CONFIG.LOGS_PATH, checkpoint_dir])

    resume_checkpoint = CONFIG.resume_model_path
    if resume_checkpoint:
        if not resume_checkpoint.endswith(".ckpt"):
            raise ValueError("--resume_model_path must point to a Lightning .ckpt checkpoint")
        if not os.path.isfile(resume_checkpoint):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint}")
        print(f" - Resuming Tetrio training from {resume_checkpoint}")

    print_separator("Loading Configuration", sep_type="SHORT")
    CONFIG.print_config()
    T_CONFIG.print_config()

    train_loader, _, val_loader = load_precomputed_tetrio_data(CONFIG)

    print(" - Loading Environment")
    observation_space = TetrisEnv(CONFIG, T_CONFIG).observation_space

    print(" - Loading Model")
    model = TurboMinoModule(CONFIG, T_CONFIG, observation_space)

    # ========================= STAGE 1: Train Classifier Only =========================
    print_separator("Train Classifier", sep_type="SUPER")
    
    epoch_checkpoint = pl.callbacks.ModelCheckpoint(
        save_top_k=-1,
        save_last=True,
        save_weights_only=False,
        every_n_epochs=1,
        filename=f"pretrain-{CONFIG.exp_name}-epoch{{epoch:02d}}",
        dirpath=checkpoint_dir,
    )
    best_checkpoint = pl.callbacks.ModelCheckpoint(
        monitor="val/acc_top10",
        mode="max",
        save_top_k=1,
        save_weights_only=False,
        filename=f"pretrain-{CONFIG.exp_name}-best",
        dirpath=checkpoint_dir,
    )
    callbacks = [
        TQDMProgressBar(refresh_rate=1),
        pl.callbacks.EarlyStopping(monitor="val/acc_top10", mode="max", patience=CONFIG.patience, verbose=True),
        epoch_checkpoint,
        best_checkpoint,
    ]

    logger = pl.loggers.CSVLogger(save_dir=CONFIG.LOGS_PATH, name=CONFIG.exp_name)

    trainer = pl.Trainer(
        max_epochs=CONFIG.tetrio_epochs, callbacks=callbacks, logger=logger,
        check_val_every_n_epoch=1, log_every_n_steps=1, deterministic="warn",
        precision="bf16-mixed",
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=resume_checkpoint,
    )

    CONFIG.final_model_path = best_checkpoint.best_model_path
    test_model(CONFIG, T_CONFIG)
