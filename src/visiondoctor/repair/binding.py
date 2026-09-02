"""The project a case is about, and the commands it declares for re-running itself."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectBinding:
    """A repository at a revision, plus how the project says to run itself.

    The commands come from the project, not from the product: nothing here knows
    what this particular cell does.
    """

    repository: Path
    revision: str
    #: Template with {input}, {output} and {log} placeholders.
    replay_command: tuple[str, ...]
    test_command: tuple[str, ...] | None = None

    def replay_command_for(self, source: Path, target: Path, log: Path) -> tuple[str, ...]:
        replacements = {"{input}": str(source), "{output}": str(target), "{log}": str(log)}
        return tuple(replacements.get(part, part) for part in self.replay_command)
