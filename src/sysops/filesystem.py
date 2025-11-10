from datetime import datetime
import git
import logging
import os
from pathlib import Path
import secrets
from typing import Union
import yaml

from src.sysops.names import *

logger = logging.getLogger(__name__)


def get_repo_root():
    """
    Get the root directory of the git repository. Fallback to current working directory if not a git repo

    Returns:
        str: The absolute path to the root directory of the git repository.
    """
    try:
        repo = git.Repo(search_parent_directories=True)
        return repo.git.rev_parse("--show-toplevel")
    except git.exc.InvalidGitRepositoryError:
        return os.getcwd()


def load_prompt_template(score_str: str) -> dict:
    template_dir = Path(get_repo_root()) / "src" / "prompts"
    template_file = template_dir / f"{score_str}_template.yaml"
    if not template_file.exists():
        raise FileNotFoundError(f"Template file not found: {template_file}")
    with open(template_file, "r") as file:
        template_dict = yaml.safe_load(file)
    return template_dict


def load_system_prompt() -> str:
    template_file = Path(get_repo_root()) / "src" / "prompts" / "system_prompt.yaml"
    if not template_file.exists():
        raise FileNotFoundError(f"Role system template file not found: {template_file}")
    with open(template_file, "r") as file:
        template_dict = yaml.safe_load(file)
    return template_dict.get("content", "")


def setup_directories(cfg):
    """
    Set up base directories for results and logs as specified in the
    configuration. If not given, use git repo or calling script directory and use default naming.

    Args:
        cfg (dict): Configuration dictionary containing directory settings.
    """
    root_dir = get_repo_root()
    output_dir = cfg.get("output_dir", "output")
    score = get_score_str(cfg["risk_score"])
    # Ensure directories are absolute paths
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(root_dir, output_dir)

    # Concatenate with run name
    run_name_default = secrets.token_hex(4)
    run_name = cfg.get("run_name", run_name_default)
    output_dir = os.path.join(output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    cfg["output_dir"] = output_dir

    # Results directories
    stage1_dir = os.path.join(output_dir, "stage1", score)
    stage2_dir = os.path.join(output_dir, "stage2", score)
    os.makedirs(stage1_dir, exist_ok=True)
    os.makedirs(stage2_dir, exist_ok=True)
    cfg["stage1_dir"] = stage1_dir
    cfg["stage2_dir"] = stage2_dir
    logger.info(
        f"Results directories created:\n\tStage 1 results: {stage1_dir}\n\tStage 2 results: {stage2_dir}"
    )

    # Log directory
    log_dir = os.path.join(output_dir, "logs", score)
    os.makedirs(log_dir, exist_ok=True)
    cfg["log_dir"] = log_dir
    logger.info(f"Logs are written to: {log_dir}")

    return cfg


def read_text_files(data_folder: Union[Path, str]) -> dict:
    assert os.path.isdir(data_folder)
    texts = {}
    for p in Path(data_folder).iterdir():
        if p.name.startswith(".") or not p.name.lower().endswith(".txt"):
            continue
        with open(p, "r") as f:
            text_id = p.stem.split("_")[0]
            texts[text_id] = f.read()
    return texts


def save_log(item: str, log_dir: Path, response: str, ts=None):
    if ts is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(log_dir / Path(f"{item}_{ts}.log"), "w") as f:
        f.write(response)
