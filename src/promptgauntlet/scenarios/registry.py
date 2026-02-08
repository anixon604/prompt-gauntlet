"""Scenario registry: discover and register scenarios."""

from __future__ import annotations

from typing import TYPE_CHECKING

from promptgauntlet.scenarios.base import ScenarioInfo

if TYPE_CHECKING:
    from promptgauntlet.scenarios.base import Scenario

# Module-level registry
_SCENARIO_CLASSES: dict[str, type[Scenario]] = {}


def register_scenario(cls: type[Scenario]) -> type[Scenario]:
    """Decorator to register a scenario class."""
    # Instantiate temporarily to read config
    instance = cls()
    _SCENARIO_CLASSES[instance.config.id] = cls
    return cls


def _ensure_loaded() -> None:
    """Import all scenario modules to trigger registration."""
    if _SCENARIO_CLASSES:
        return
    # Import each family's scenario module
    import promptgauntlet.scenarios.classification.scenario  # noqa: F401
    import promptgauntlet.scenarios.constraint.scenario  # noqa: F401
    import promptgauntlet.scenarios.convergence.scenario  # noqa: F401
    import promptgauntlet.scenarios.tool_use.scenario  # noqa: F401


class ScenarioRegistry:
    """Provides access to registered scenarios."""

    def list_scenarios(self, family: str | None = None) -> list[ScenarioInfo]:
        """List all registered scenarios, optionally filtered by family."""
        _ensure_loaded()
        infos: list[ScenarioInfo] = []
        for cls in _SCENARIO_CLASSES.values():
            instance = cls()
            cfg = instance.config
            if family and cfg.family.value != family:
                continue
            infos.append(
                ScenarioInfo(
                    id=cfg.id,
                    family=cfg.family,
                    name=cfg.name,
                    description=cfg.description,
                )
            )
        return sorted(infos, key=lambda s: (s.family.value, s.id))

    def get_scenario(self, scenario_id: str) -> Scenario:
        """Get a scenario instance by ID."""
        _ensure_loaded()
        cls = _SCENARIO_CLASSES.get(scenario_id)
        if cls is None:
            raise KeyError(f"Unknown scenario: {scenario_id}")
        return cls()

    def get_all_scenario_ids(self) -> list[str]:
        """Return all registered scenario IDs."""
        _ensure_loaded()
        return sorted(_SCENARIO_CLASSES.keys())

    def resolve_scenario_ids(self, patterns: list[str]) -> list[str]:
        """Resolve scenario globs/patterns to concrete IDs.

        Supports:
        - 'all' -> all scenarios
        - 'classification/*' -> all in family
        - exact ID
        """
        import fnmatch

        _ensure_loaded()
        all_ids = self.get_all_scenario_ids()
        if patterns == ["all"] or "all" in patterns:
            return all_ids
        matched: list[str] = []
        for pattern in patterns:
            if pattern in all_ids:
                matched.append(pattern)
            else:
                for sid in all_ids:
                    if fnmatch.fnmatch(sid, pattern):
                        matched.append(sid)
        return sorted(set(matched))


# Singleton
_registry = ScenarioRegistry()


def get_registry() -> ScenarioRegistry:
    """Return the global scenario registry."""
    return _registry
