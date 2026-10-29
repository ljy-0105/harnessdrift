"""`init` 不能在没有明确授权时清空已有目录。"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from harnessdrift.cli import cmd_init


def _tree(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "systemprompt.md").write_text("# source\n", encoding="utf-8")


def test_init_refuses_a_nonempty_destination(tmp_path: Path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    _tree(source)
    destination.mkdir()
    (destination / "keep.txt").write_text("do not delete\n", encoding="utf-8")

    rc = cmd_init(Namespace(harness=str(source), dest=str(destination), force=False))

    assert rc == 2
    assert (destination / "keep.txt").read_text(encoding="utf-8") == "do not delete\n"


def test_init_replaces_only_with_force(tmp_path: Path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    _tree(source)
    destination.mkdir()
    (destination / "old.txt").write_text("old\n", encoding="utf-8")

    rc = cmd_init(Namespace(harness=str(source), dest=str(destination), force=True))

    assert rc == 0
    assert not (destination / "old.txt").exists()
    assert (destination / "systemprompt.md").exists()
