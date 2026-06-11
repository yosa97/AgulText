DEFAULT_IMAGE_DOCKERFILE_PATH = "dockerfiles/standalone-image-trainer.dockerfile"
DEFAULT_TEXT_DOCKERFILE_PATH = "dockerfiles/standalone-text-trainer.dockerfile"
TRAINER_CHECKPOINTS_PATH = "/tmp/trainer/checkpoints"
TEMP_REPO_PATH = "/tmp/trainer/repos/"
TASKS_FILE_PATH = "trainer/task_history.json"
VOLUME_NAMES = ["checkpoints", "cache"]
HF_UPLOAD_DOCKER_IMAGE = "diagonalge/hf-uploader:latest"
TRAINER_DOWNLOADER_DOCKER_IMAGE = "diagonalge/trainer-downloader:latest"
CACHE_CLEANER_DOCKER_IMAGE = "diagonalge/cache-cleaner:latest"
IMAGE_TASKS_HF_SUBFOLDER_PATH = "checkpoints"
DEFAULT_TRAINING_CONTAINER_MEM_LIMIT = "24g"
DEFAULT_TRAINING_CONTAINER_NANO_CPUS = 8

# Dynamic resource allocation based on GPU count
# For 8xH100 with 1440GB RAM and 252 CPUs
MEMORY_PER_GPU_GB = 135  # 75% of 1440GB / 8 GPUs
CPUS_PER_GPU = 24  # Conservative allocation leaving headroom

CACHE_CLEANUP_CUTOFF_HOURS = 12
STALE_TASK_GRACE_MINUTES = 10

#TRAINING PATHS 
CACHE_ROOT_PATH = "/cache"
HUGGINGFACE_CACHE_PATH = "/cache/hf_cache"
OUTPUT_CHECKPOINTS_PATH = "/app/checkpoints/"
CACHE_MODELS_DIR = "/cache/models"
CACHE_DATASETS_DIR = "/cache/datasets"
WANDB_LOGS_DIR = "/cache/wandb_logs"
IMAGE_CONTAINER_CONFIG_TEMPLATE_PATH = "/workspace/core/config"
IMAGE_CONTAINER_CONFIG_SAVE_PATH = "/dataset/configs"
IMAGE_CONTAINER_IMAGES_PATH = "/dataset/images"
TEXT_CONTAINER_SAVE_PATH = "/workspace/axolotl/outputs/"

#Directories

AXOLOTL_DIRECTORIES = {
    "data": "/workspace/axolotl/data",
    "prepared": "/workspace/axolotl/data_prepared",
    "configs": "/workspace/axolotl/configs",
    "outputs": "/workspace/axolotl/outputs",
    "input": "/workspace/input_data",
    "root": "/workspace/axolotl",
    "src": "/workspace/axolotl/src/"
}

WANDB_DIRECTORIES = [
    "WANDB_DIR",
    "WANDB_CACHE_DIR",
    "WANDB_ARTIFACT_DIR",
    "WANDB_DATA_DIR",
    "WANDB_CONFIG_DIR",
]

# ── AgulText training-quality constants ───────────────────────────────────────
# How often (in steps) to emit GPU memory / grad-norm diagnostics
PERF_MONITOR_INTERVAL_STEPS = 50

# Maximum number of checkpoint shards to retain on disk before pruning
CHECKPOINT_KEEP_COUNT = 3

# Salt added to the task_id SHA-256 seed used by add_random_noise.py so
# the resulting noise vector is unique to this miner instance.
NOISE_SEED_SALT = "agultext_v2"

# Minimum number of gradient steps before an eval checkpoint is saved
MIN_EVAL_INTERVAL_STEPS = 25

# Staging sub-directory used for in-progress checkpoint uploads
SUBMISSION_STAGING_SUBDIR = "staging"

# Grace period (seconds) before declaring a stale download dead
DOWNLOAD_STALE_TIMEOUT_SECONDS = 900
