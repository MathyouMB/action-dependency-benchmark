import os

from benchmark.env import load_env_file


def test_load_env_file_sets_variables_from_the_file(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("OPENROUTER_API_KEY=sk-or-from-file\n")

    load_env_file(env)

    assert os.environ["OPENROUTER_API_KEY"] == "sk-or-from-file"


def test_load_env_file_skips_blank_lines_and_comments(tmp_path, monkeypatch):
    monkeypatch.delenv("A_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("\n# a comment with = in it\n\nA_KEY=value\n")

    load_env_file(env)

    assert os.environ["A_KEY"] == "value"


def test_load_env_file_strips_surrounding_quotes(tmp_path, monkeypatch):
    monkeypatch.delenv("A_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text('A_KEY="quoted value"\n')

    load_env_file(env)

    assert os.environ["A_KEY"] == "quoted value"


def test_load_env_file_does_not_clobber_an_exported_variable(tmp_path, monkeypatch):
    monkeypatch.setenv("A_KEY", "from-the-shell")
    env = tmp_path / ".env"
    env.write_text("A_KEY=from-the-file\n")

    load_env_file(env)

    assert os.environ["A_KEY"] == "from-the-shell"


def test_load_env_file_is_a_no_op_when_there_is_no_file(tmp_path):
    load_env_file(tmp_path / "absent.env")
