import pytorch_lightning as pl

from maikol_utils.print_utils import print_separator
from maikol_utils.file_utils import make_dirs

from src.data import load_tetrio_data
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
    make_dirs([CONFIG.LOGS_PATH, CONFIG.pretrain_model_path])

    print_separator("Loading Configuration", sep_type="SHORT")
    CONFIG.print_config()
    T_CONFIG.print_config()

    train_loader, _, val_loader = load_tetrio_data(CONFIG, T_CONFIG)

    print(" - Loading Environment")
    observation_space = TetrisEnv(CONFIG, T_CONFIG).observation_space

    print(" - Loading Model")
    model = TurboMinoModule(CONFIG, T_CONFIG, observation_space)

    # ========================= STAGE 1: Train Classifier Only =========================
    print_separator("Train Classifier", sep_type="SUPER")
    
    callbacks = [
        pl.callbacks.EarlyStopping(monitor="val_acc", mode="max", patience=CONFIG.patience, verbose=True),
        pl.callbacks.ModelCheckpoint(
            monitor="val_acc", mode="max", save_top_k=1, save_weights_only=True, 
            filename=f"pretrain-{CONFIG.exp_name}-{{epoch:02d}}-{{val_acc:.4f}}", 
            dirpath=CONFIG.pretrain_model_path
        ),
    ]

    logger = pl.loggers.CSVLogger(save_dir=CONFIG.LOGS_PATH, name=CONFIG.exp_name)

    trainer = pl.Trainer(
        max_epochs=CONFIG.epochs, callbacks=callbacks, logger=logger,
        check_val_every_n_epoch=1, log_every_n_steps=1, deterministic="warn"
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


    CONFIG.final_model_path = trainer.checkpoint_callbacks[0].best_model_path
    test_model(CONFIG, T_CONFIG)
