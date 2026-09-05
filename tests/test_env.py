from pathlib import Path

from ml_for_malaria.env import load_local_env, parse_env_file


def test_parse_env_file(tmp_path: Path):
    path = tmp_path / ".local.env"
    path.write_text(
        "# comment\n"
        "TABPFN_TOKEN=secret\n"
        "QUOTED='value'\n"
        "EMPTY=\n"
        "\n"
        "MONROE_HOME=/tmp/monroe\n",
        encoding="utf-8",
    )
    values = parse_env_file(path)
    assert values["TABPFN_TOKEN"] == "secret"
    assert values["QUOTED"] == "value"
    assert values["EMPTY"] == ""
    assert values["MONROE_HOME"] == "/tmp/monroe"


def test_load_local_env_does_not_override_process(tmp_path: Path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    env_path = root / ".local.env"
    env_path.write_text("TABPFN_TOKEN=from_file\nOTHER=file\n", encoding="utf-8")
    monkeypatch.setenv("TABPFN_TOKEN", "from_shell")
    monkeypatch.delenv("OTHER", raising=False)
    monkeypatch.setattr("ml_for_malaria.env._LOADED", False)
    loaded = load_local_env(force=True, start=root)
    assert loaded == env_path
    assert Path.cwd()  # keep pytest happy if cwd tools run
    import os

    assert os.environ["TABPFN_TOKEN"] == "from_shell"
    assert os.environ["OTHER"] == "file"
