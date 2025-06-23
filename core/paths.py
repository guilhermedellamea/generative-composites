from pathlib import Path

# Please configure these paths to your data directories:
BASE_DIR = Path(__file__).resolve().parent.parent
BASE_DATA_DIR = BASE_DIR / "data"


OUTPUT_FOLDER = BASE_DIR / "output"
SAVED_MODELS_FOLDER = OUTPUT_FOLDER / "saved_models"
FIGURES_FOLDER = OUTPUT_FOLDER / "figures"


def get_saved_models_path(name: str) -> Path:
    SAVED_MODELS_FOLDER.mkdir(parents=True, exist_ok=True)

    file_path = SAVED_MODELS_FOLDER / f"{name}"

    return file_path


def get_saved_figure_path(name: str) -> Path:
    FIGURES_FOLDER.mkdir(parents=True, exist_ok=True)

    file_path = FIGURES_FOLDER / f"{name}.png"

    return file_path


def get_gradcam_folder(dataset: str) -> Path:
    gradcam_folder = BASE_DATA_DIR / dataset / "gradcam"
    gradcam_folder.mkdir(parents=True, exist_ok=True)
    return gradcam_folder
