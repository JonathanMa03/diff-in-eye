from pathlib import Path
import shutil
import kagglehub

# Download dataset from Kaggle
path = kagglehub.dataset_download(
    "obulisainaren/retinal-oct-c8"
)

print(f"Downloaded dataset to: {path}")

# Move dataset into project data/raw directory
source_path = Path(path)
target_path = Path("data/raw/retinal-oct-c8")

target_path.parent.mkdir(parents=True, exist_ok=True)

# Copy dataset into repository
if not target_path.exists():
    shutil.copytree(source_path, target_path)

print(f"Dataset copied to: {target_path}")