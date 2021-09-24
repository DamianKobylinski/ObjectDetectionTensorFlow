import os
import config as cfg

if not os.path.exists(cfg.files['TF_RECORD_SCRIPT']):
    os.system(f"git clone https://github.com/nicknochnack/GenerateTFRecord {cfg.paths['SCRIPTS_PATH']}")

os.system(
    f"python {cfg.files['TF_RECORD_SCRIPT']} "
    f"-x {os.path.join(cfg.paths['IMAGE_PATH'], 'train')} "
    f"-l {cfg.files['LABELMAP']} "
    f"-o {os.path.join(cfg.paths['ANNOTATION_PATH'], 'train.record')}")
os.system(
    f"python {cfg.files['TF_RECORD_SCRIPT']} "
    f"-x {os.path.join(cfg.paths['IMAGE_PATH'], 'test')} "
    f"-l {cfg.files['LABELMAP']} "
    f"-o {os.path.join(cfg.paths['ANNOTATION_PATH'], 'test.record')}")

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
