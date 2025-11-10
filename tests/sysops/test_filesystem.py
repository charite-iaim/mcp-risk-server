# mcp/tests/sysops/test_filesystem.py

import pytest
import os
from pathlib import Path
import types


from src.sysops import filesystem


@pytest.fixture
def scores_default():
    return {"hasbled", "cha2ds2vasc", "euroscoreii"}

def testget_repo_root_git(monkeypatch):
    monkeypatch.setattr(
        filesystem.git, 
        "Repo", 
        lambda **kwargs: types.SimpleNamespace(
            git=types.SimpleNamespace(rev_parse=lambda x: "/fake/repo/root")
        )
    )
    print(filesystem.get_repo_root())
    assert filesystem.get_repo_root() == "/fake/repo/root"

def testget_repo_root_not_git(monkeypatch):
    monkeypatch.setattr(filesystem.git, "Repo", lambda **kwargs: (_ for _ in ()).throw(filesystem.git.exc.InvalidGitRepositoryError()))
    monkeypatch.setattr(os, "getcwd", lambda: "/fake/cwd")
    assert filesystem.get_repo_root() == "/fake/cwd"

def test_load_prompt_template_presence(scores_default):
    root_dir = Path(filesystem.get_repo_root())
    template_dir = root_dir / "src" / "prompts"
    scores_detected = set()
    for fname in os.listdir(template_dir):
        name, ext = os.path.splitext(fname)
        if name.endswith("_template") and ext in [".yaml", ".yml"]:
            scores_detected.add(name.split("_template")[0])
    assert scores_detected == scores_default

def test_load_prompt_templates(scores_default):
    for score in scores_default:
        td = filesystem.load_prompt_template(score_str=score)
        assert len(td) > 10

def test_load_prompt_template_not_found(tmp_path, monkeypatch):
    monkeypatch.setattr(filesystem, "__file__", str(tmp_path / "happy_template.yaml"))
    monkeypatch.setattr(filesystem.Path, "parent", property(lambda self: tmp_path))
    with pytest.raises(FileNotFoundError):
        filesystem.load_prompt_template("missing")

def test_setup_directories_creates_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(filesystem, "get_repo_root", lambda: str(tmp_path))
    monkeypatch.setattr(filesystem, "get_str_representation", lambda x: "happy_score")
    cfg = {"risk_score": 'happy_score'}
    updated_cfg = filesystem.setup_directories(cfg)
    output_dir = updated_cfg["output_dir"]
    assert os.path.isdir(output_dir)
    assert os.path.isdir(updated_cfg["stage1_dir"])
    assert os.path.isdir(updated_cfg["stage2_dir"])
    assert os.path.isdir(updated_cfg["log_dir"])
    

def test_setup_directories_with_output_folder_and_run_name(tmp_path, monkeypatch):
    monkeypatch.setattr(filesystem, "get_repo_root", lambda: str(tmp_path))
    monkeypatch.setattr(filesystem, "get_str_representation", lambda x: "happyness")
    output_dir1 = Path(tmp_path) / "my_output"
    run_name = "my_run"
    output_dir2 = output_dir1 / run_name
    cfg = {"risk_score": "happyness", "output_dir": output_dir1, "run_name": run_name}
    updated_cfg = filesystem.setup_directories(cfg)
    assert str(output_dir2) == updated_cfg["output_dir"]
    assert run_name in updated_cfg["output_dir"]
    assert cfg["risk_score"] in cfg["stage1_dir"]
    assert cfg["risk_score"] in cfg["stage2_dir"]
    assert "stage1" in cfg["stage1_dir"]
    assert "stage2" in cfg["stage2_dir"]
    assert cfg["risk_score"] in cfg["log_dir"]
    

def test_read_text_files_tmp(tmp_path):
    file1 = tmp_path / "a.txt"
    file2 = tmp_path / "b.txt"
    file3 = tmp_path / "ignore.md"
    file1.write_text("hello")
    file2.write_text("world")
    file3.write_text("not a txt")
    result = filesystem.read_text_files(str(tmp_path))
    assert result == {"a": "hello", "b": "world"}

def test_read_text_files_test_data():
    root_dir = Path(filesystem.get_repo_root())
    for score in ['cha2ds2vasc', 'euroscoreii', 'hasbled']:
        dir_score = root_dir / "tests" / "data" / f"reports_{score}"
        texts = filesystem.read_text_files(dir_score)
        assert isinstance(texts, dict)
        assert len(texts) == 2
        pids = set()
        for pid, text in texts.items():
            assert isinstance(pid, str)
            assert isinstance(text, str)
            assert len(text) > 100
            assert pid not in pids
            pids.add(pid)
