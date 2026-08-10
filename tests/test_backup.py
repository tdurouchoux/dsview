import os
from dataclasses import dataclass
from pathlib import Path

import pytest
import typer

from dsview import cli
from dsview.obsidian.sync_vault import CommandFailed


@dataclass
class FakePostgresConfig:
    host: str = "localhost"
    port: int = 5432
    database: str = "test_db"
    user: str = "test_user"
    password: str = "test_password"


@pytest.fixture
def fake_pg_config(monkeypatch):
    monkeypatch.setattr(cli, "load_postgres_config", lambda: FakePostgresConfig())


@pytest.fixture
def fake_pg_tools(monkeypatch):
    monkeypatch.setattr(
        cli.shutil,
        "which",
        lambda tool: f"/usr/bin/{tool}" if tool.startswith("pg_") else None,
    )


@pytest.fixture
def captured_run_cmd(monkeypatch):
    calls = []

    def fake_run_cmd(cmd, label, **kwargs):
        calls.append((cmd, label, kwargs))

    monkeypatch.setattr(cli, "run_cmd", fake_run_cmd)
    return calls


def test_backup_invokes_pg_dump_with_timestamped_output(
    tmp_path, monkeypatch, fake_pg_config, fake_pg_tools, captured_run_cmd
):
    monkeypatch.chdir(tmp_path)

    cli.backup()

    assert len(captured_run_cmd) == 1
    cmd, label, kwargs = captured_run_cmd[0]

    assert label == "pg_dump backup"
    assert cmd[0] == "/usr/bin/pg_dump"
    assert "-Fc" in cmd
    assert "--no-owner" in cmd
    assert "--no-privileges" in cmd
    assert cmd[cmd.index("-h") + 1] == "localhost"
    assert cmd[cmd.index("-p") + 1] == "5432"
    assert cmd[cmd.index("-U") + 1] == "test_user"
    assert cmd[cmd.index("-d") + 1] == "test_db"

    output_file = Path(cmd[cmd.index("-f") + 1])
    assert output_file.resolve().parent == tmp_path / "backup"
    assert output_file.resolve().parent.exists()
    assert output_file.name.startswith("dsview_")
    assert output_file.suffix == ".dump"

    assert "test_password" not in cmd
    assert kwargs["env"]["PGPASSWORD"] == "test_password"


def test_backup_raises_when_pg_dump_missing(
    tmp_path, monkeypatch, fake_pg_config, captured_run_cmd
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli.shutil, "which", lambda tool: None)

    with pytest.raises(cli.MissingPostgresTool):
        cli.backup()

    assert captured_run_cmd == []


def test_backup_propagates_command_failure(
    tmp_path, monkeypatch, fake_pg_config, fake_pg_tools
):
    monkeypatch.chdir(tmp_path)

    def failing_run_cmd(cmd, label, **kwargs):
        raise CommandFailed(label, "", "pg_dump: error")

    monkeypatch.setattr(cli, "run_cmd", failing_run_cmd)

    with pytest.raises(CommandFailed):
        cli.backup()


def test_restore_db_picks_newest_dump_and_confirms(
    tmp_path, monkeypatch, fake_pg_config, fake_pg_tools, captured_run_cmd
):
    backup_dir = tmp_path / "backup"
    backup_dir.mkdir()

    older = backup_dir / "dsview_20260101_000000.dump"
    newer = backup_dir / "dsview_20260810_000000.dump"
    older.write_bytes(b"old")
    newer.write_bytes(b"new")

    now = 1_700_000_000
    os.utime(older, (now, now))
    os.utime(newer, (now + 100, now + 100))

    monkeypatch.setattr(typer, "confirm", lambda *a, **k: True)

    cli.restore_db(backup_file=None, backup_dir=str(backup_dir))

    assert len(captured_run_cmd) == 1
    cmd, label, kwargs = captured_run_cmd[0]

    assert label == "pg_restore restore"
    assert cmd[0] == "/usr/bin/pg_restore"
    assert "--clean" in cmd
    assert "--if-exists" in cmd
    assert str(newer) in cmd
    assert str(older) not in cmd
    assert kwargs["env"]["PGPASSWORD"] == "test_password"


def test_restore_db_aborts_when_not_confirmed(
    tmp_path, monkeypatch, fake_pg_config, fake_pg_tools, captured_run_cmd
):
    backup_dir = tmp_path / "backup"
    backup_dir.mkdir()
    (backup_dir / "dsview_20260810_000000.dump").write_bytes(b"data")

    monkeypatch.setattr(typer, "confirm", lambda *a, **k: False)

    with pytest.raises(typer.Abort):
        cli.restore_db(backup_file=None, backup_dir=str(backup_dir))

    assert captured_run_cmd == []


def test_restore_db_raises_when_no_dump_files(tmp_path, fake_pg_config):
    backup_dir = tmp_path / "backup"
    backup_dir.mkdir()

    with pytest.raises(cli.NoBackupFileFound):
        cli.restore_db(backup_file=None, backup_dir=str(backup_dir))


def test_restore_db_raises_when_backup_dir_missing(tmp_path, fake_pg_config):
    with pytest.raises(cli.MissingBackupDirectory):
        cli.restore_db(backup_file=None, backup_dir=str(tmp_path / "does_not_exist"))
