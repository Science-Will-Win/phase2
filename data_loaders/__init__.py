"""
Dataset routing package.

Routes --dataset_type to the corresponding module inside data_loaders/.
Each module must expose: get_dataset(args, tokenizer) -> (train_dataset, eval_dataset)
"""

import importlib
import importlib.util
import os


def get_dataset(args, tokenizer):
    """
    Dynamically load the dataset module specified by args.dataset_type
    and delegate to its get_dataset() function.

    args.dataset_type can be:
      - A name   (e.g. "instruction_dataset")  -> data_loaders/instruction_dataset.py
      - A path   (e.g. "custom/my_dataset")    -> custom/my_dataset.py
    The .py suffix is optional in both cases.
    """
    dataset_type = args.dataset_type

    if dataset_type.endswith(".py"):
        dataset_type = dataset_type[:-3]

    if os.sep in dataset_type or "/" in dataset_type:
        file_path = dataset_type if dataset_type.endswith(".py") else dataset_type + ".py"
        if not os.path.isfile(file_path):
            raise FileNotFoundError(
                f"Dataset module not found at path: {file_path}"
            )
        spec = importlib.util.spec_from_file_location("custom_dataset", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        try:
            module = importlib.import_module(f"data_loaders.{dataset_type}")
        except ModuleNotFoundError:
            available = _list_available()
            raise ModuleNotFoundError(
                f"Dataset type '{dataset_type}' not found in data_loaders/. "
                f"Available: {available}"
            )

    if not hasattr(module, "get_dataset"):
        raise AttributeError(
            f"Dataset module '{dataset_type}' must define a "
            f"get_dataset(args, tokenizer) function."
        )

    return module.get_dataset(args, tokenizer)


def _list_available():
    """List dataset module names available in data_loaders/."""
    pkg_dir = os.path.dirname(__file__)
    names = []
    for fname in sorted(os.listdir(pkg_dir)):
        if fname.endswith(".py") and fname != "__init__.py":
            names.append(fname[:-3])
    return names
