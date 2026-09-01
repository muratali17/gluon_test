import os

TRAINED_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "trained_models")


def validate_task_name(task_name):
    if not task_name or task_name.strip() != task_name or '/' in task_name or '\\' in task_name or task_name in {'.', '..'}:
        raise ValueError(f"Invalid task_name: {task_name!r}")


def get_model_path(task_name):
    validate_task_name(task_name)
    return os.path.join(TRAINED_MODELS_DIR, task_name)
