import os
import config as cfg

TRAINING_SCRIPT = os.path.join(
    cfg.paths['APIMODEL_PATH'],
    'research',
    'object_detection',
    'model_main_tf2.py')

os.system(
    f"python {TRAINING_SCRIPT} "
    f"--model_dir={cfg.paths['CHECKPOINT_PATH']} "
    f"--pipeline_config_path={cfg.files['PIPELINE_CONFIG']} "
    f"--num_train_steps=2000")

os.system(
    f"python {TRAINING_SCRIPT} "
    f"--model_dir={cfg.paths['CHECKPOINT_PATH']} "
    f"--pipeline_config_path={cfg.files['PIPELINE_CONFIG']} "
    f"--checkpoint_dir={cfg.paths['CHECKPOINT_PATH']}")
