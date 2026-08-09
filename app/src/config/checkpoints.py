"""Checkpoint naming and resume metadata constants."""

RESUME_STATE_SUFFIX = ".resume.yaml"
RESUME_STATE_VERSION = 1
BEST_MODEL_FILENAME = "best_model.zip"
DEFAULT_CHECKPOINT_PREFIX = "turbomino_ckpt"
STAGE_CHECKPOINT_PREFIX = "turbomino_{stage_label}_ckpt"
STAGE_MODEL_FILENAME = "tetris_turbomino_{exp_name}_{stage_label}.zip"

IGNORED_RESUME_CONFIG_FIELDS = frozenset({
    "config",
    "model_path",
    "resume_model_path",
    "DATA_PATH",
    "MODELS_PATH",
    "LOGS_PATH",
    "CONFIGS_PATH",
    "raw_dataset_path",
    "processed_dataset_path",
    "tetrio_train",
    "tetrio_test",
    "tetrio_val",
    "pretrain_model_path",
    "checkpoint_dir",
    "log_dir",
    "final_model_path",
    "best_model_path",
    "lr_end",
    "ent_coef_end",
    "reset_stage",
})
