"""Prompt loading.

Prompts live as editable files under ./prompts, never as string literals in code
(CLAUDE.md §4.2), so they can be tuned from the confusion report without a code change.

Substitution is deliberately restricted to an explicit allowlist of context keys. A
prompt cannot interpolate a filename or a keyword because nothing ever passes those
in — see images.staged_pixels for the matching guarantee on the pixel side.
"""

from __future__ import annotations

from pathlib import Path
from string import Template

ROUTING_PROMPT = "taxon_routing"
SPORT_ROUTING_PROMPT = "sport_routing"

#: Which routing prompt each profile uses.
ROUTING_FOR_PROFILE = {
    "wildlife": ROUTING_PROMPT,
    "sport": SPORT_ROUTING_PROMPT,
}

# Only these may ever be substituted into a prompt. Anything else raises.
ALLOWED_CONTEXT_KEYS = frozenset({"season_context", "location_context"})


class PromptError(RuntimeError):
    pass


class PromptLibrary:
    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        if not self.directory.is_dir():
            raise PromptError(f"Prompt directory not found: {self.directory}")

    def path_for(self, name: str) -> Path:
        return self.directory / f"{name}.md"

    def available(self) -> list[str]:
        return sorted(p.stem for p in self.directory.glob("*.md"))

    def render(self, name: str, **context: str) -> str:
        unexpected = set(context) - ALLOWED_CONTEXT_KEYS
        if unexpected:
            raise PromptError(
                f"Refusing to render prompt with disallowed context keys: {sorted(unexpected)}. "
                "Only pixels and approved context may reach the model."
            )
        file = self.path_for(name)
        if not file.is_file():
            raise PromptError(f"No prompt file for '{name}' at {file}")
        template = Template(file.read_text("utf-8"))
        filled = {key: context.get(key, "") for key in ALLOWED_CONTEXT_KEYS}
        return template.safe_substitute(filled).strip()

    def prompt_for_taxon(self, taxon: str) -> str:
        """Taxon-specific prompt, falling back to the generic one."""
        name = taxon if self.path_for(taxon).is_file() else "generic"
        return self.render(name)

    def fingerprint(self) -> str:
        """Hash of every prompt file's contents.

        Part of the cache key: editing a prompt must invalidate results produced by
        the previous wording, otherwise a tuning pass silently re-serves stale
        identifications and looks like it worked.
        """
        import hashlib

        digest = hashlib.sha256()
        for path in sorted(self.directory.glob("*.md")):
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
        return digest.hexdigest()[:16]
