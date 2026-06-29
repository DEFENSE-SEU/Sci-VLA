"""Per-dataset Python schema registry — replaces meta/labvla_manifest.json.

A schema file is any Python module that either:
  (a) defines a module-level `SCHEMA: DatasetSchema`, or
  (b) calls `register(SCHEMA, name=..., aliases=...)` at import time.

Resolution order for `--dataset_schema <spec>`:
  1. Registry lookup by exact name (populated by `register()` or by the
     lazy auto-import of the in-repo `schemas/` package).
  2. Dotted Python path  `pkg.module:ATTR`  (optional `:ATTR`; defaults to `SCHEMA`).
  3. Absolute file path  `/abs/path/schema.py[:ATTR]`.
  4. Fail with fuzzy suggestions from the registry.

A register-only module (one that sets `SCHEMA = None` / omits `SCHEMA` and
instead `register(...)`s by name at import time — e.g. schemas/labutopia_level3.py)
is supported by BOTH the dotted-path and file-path forms: importing the module
runs its `register()` side effects, and resolution then falls back to a
registry lookup for `<spec>` rather than raising on the missing/None `SCHEMA`.
Pass `:ATTR` only when the module truly exposes that attribute.

The returned `DatasetSchema` is passed to `discover_schema(..., override=...)`
which is then threaded through the adapter. Downstream (hydrate_all, deploy)
is unchanged.
"""
from __future__ import annotations

import difflib
import importlib
import importlib.util
import logging
import os
from pathlib import Path
from typing import Iterable, Optional

from .dataset_schema import DatasetSchema
from .errors import SchemaSpecError  # noqa: F401  # re-export for backward compat

logger = logging.getLogger(__name__)

# name -> DatasetSchema instance. Populated at import time by `register()`.
_REGISTRY: dict[str, DatasetSchema] = {}

# Track aliases separately so error messages can hint at the canonical name.
_ALIASES: dict[str, str] = {}

# Idempotence guard: the lazy `schemas/` auto-import runs at most once per
# process; nested calls during `resolve()` just short-circuit.
_AUTOLOADED = False

# Sibling `schemas/*.py` modules that failed to import during the in-repo
# autoload, captured as ``[(module_name, exception), ...]``. Surfaced loudly on
# the first `resolve()` / `registered_names()` so a broken schema file does not
# silently masquerade as an "unknown schema".
_AUTOLOAD_FAILURES: list[tuple[str, Exception]] = []


def _raise_if_autoload_failed() -> None:
    """Fail loud if any in-repo `schemas/*.py` failed to import during autoload.

    Splices the real underlying exception(s) into a ``SchemaSpecError`` so the
    operator sees the actual ImportError root cause instead of a misleading
    "unknown schema" miss. Only fires when there is at least one failure — a
    clean ``schemas/`` directory leaves behavior unchanged.
    """
    if not _AUTOLOAD_FAILURES:
        return
    details = "; ".join(
        f"{name}: {type(exc).__name__}: {exc}" for name, exc in _AUTOLOAD_FAILURES
    )
    first_exc = _AUTOLOAD_FAILURES[0][1]
    raise SchemaSpecError(
        f"schema autoload failed for {len(_AUTOLOAD_FAILURES)} module(s) in the "
        f"in-repo schemas/ package: {details}. Fix the import error(s) above — a "
        f"broken schema module would otherwise look like an 'unknown schema'."
    ) from first_exc


def register(
    schema: DatasetSchema,
    *,
    name: Optional[str] = None,
    aliases: Iterable[str] = (),
) -> DatasetSchema:
    """Register a DatasetSchema under `name` (default: schema.schema_id).

    Also accepts optional aliases (useful when a dataset has both a short and
    long name, e.g. `labutopia_transport_beaker` + `Level3_TransportBeaker`).
    Returns the schema unchanged so callers can write `SCHEMA = register(...)`.
    """
    if not isinstance(schema, DatasetSchema):
        raise TypeError(
            f"register() expects DatasetSchema, got {type(schema).__name__}. "
            f"Construct it via schema.presets.franka8(...) or DatasetSchema(...)."
        )
    key = name or schema.schema_id
    if key in _REGISTRY and _REGISTRY[key] is not schema:
        existing = _REGISTRY[key]
        _same_source = False
        try:
            _same_source = (existing.source_path and schema.source_path
                            and Path(str(existing.source_path)).resolve()
                            == Path(str(schema.source_path)).resolve())
        except OSError:
            pass
        _equal = False
        try:
            _equal = existing.to_dict() == schema.to_dict()
        except Exception:  # noqa: BLE001
            pass
        if _same_source or _equal:
            # Benign re-import of the same module/schema (e.g. via a second
            # path spelling) — keep quiet idempotency.
            logger.debug("register(): re-registering identical schema %r", key)
        elif os.environ.get("LABVLA_ALLOW_SCHEMA_OVERRIDE") == "1":
            logger.warning(
                "register(): OVERRIDING schema %r (was schema_id=%s from %s, "
                "now schema_id=%s from %s) — explicitly allowed via "
                "LABVLA_ALLOW_SCHEMA_OVERRIDE=1",
                key, existing.schema_id, existing.source_path,
                schema.schema_id, schema.source_path,
            )
        else:
            # Schema identity drives layout/delta-mask/camera semantics; a
            # silent last-import-wins override would make a stable
            # --dataset_schema spec depend on import order.
            raise ValueError(
                f"register(): schema name {key!r} is already registered with "
                f"DIFFERENT content (existing from {existing.source_path}, "
                f"new from {schema.source_path}). Set "
                f"LABVLA_ALLOW_SCHEMA_OVERRIDE=1 to override deliberately."
            )
    _REGISTRY[key] = schema
    for a in aliases:
        if a == key:
            continue  # self-alias is a no-op, not a collision
        if a in _REGISTRY:
            # Alias would shadow an existing canonical entry → refuse.
            # `resolve()` consults _ALIASES before _REGISTRY, so silently
            # accepting this makes the canonical schema under name `a`
            # unreachable by name.
            raise ValueError(
                f"register(): alias {a!r} collides with existing canonical "
                f"schema {a!r} (schema_id={_REGISTRY[a].schema_id}). "
                f"Pick a different alias."
            )
        if a in _ALIASES and _ALIASES[a] != key:
            logger.warning("register(): alias %r already points to %r, overwriting with %r",
                           a, _ALIASES[a], key)
        _ALIASES[a] = key
    return schema


def registered_names() -> list[str]:
    """Return canonical names known to the registry (not aliases).

    If any in-repo schema module failed to import during autoload the returned
    set would be silently incomplete, so fail loud with the real underlying
    exception instead of returning a partial list.
    """
    _ensure_autoload()
    _raise_if_autoload_failed()
    return sorted(_REGISTRY.keys())


def _ensure_autoload() -> None:
    """Lazy auto-import of the in-repo `schemas/` package + any dirs listed
    in the LABVLA_SCHEMA_PATH env var (colon-separated, PYTHONPATH-style).

    Done lazily so CLI cold-start isn't slowed down when `--dataset_schema`
    is unused, and to avoid surprising import side-effects in library code.
    """
    global _AUTOLOADED
    if _AUTOLOADED:
        return

    # Only set _AUTOLOADED=True AFTER successful completion. The
    # prior "set first, import later" meant a transient ImportError would
    # permanently disable schema discovery in the process (the next call
    # short-circuits). Now a failed import leaves the gate open for retry.

    # In-repo default package: import by name. The top-level `schemas/`
    # package's __init__.py performs the glob import of sibling files.
    _AUTOLOAD_FAILURES.clear()
    try:
        schemas_pkg = importlib.import_module("schemas")
    except ModuleNotFoundError:
        pass  # no in-repo schemas/ dir — user may rely on dotted/file specs only
    except Exception as e:  # pragma: no cover
        logger.warning("schemas/ auto-import raised %r; will retry on next call", e)
        return
    else:
        # Capture per-sibling import failures recorded by the package so
        # resolve()/registered_names() can surface them loudly.
        failures = getattr(schemas_pkg, "IMPORT_FAILURES", None)
        if failures:
            _AUTOLOAD_FAILURES.extend(failures)

    # External dirs from env.
    extra_paths = os.environ.get("LABVLA_SCHEMA_PATH", "")
    if extra_paths:
        for p in extra_paths.split(os.pathsep):
            p = p.strip()
            if not p:
                continue
            _import_all_py_in_dir(Path(p))

    _AUTOLOADED = True


def _import_all_py_in_dir(d: Path) -> None:
    if not d.is_dir():
        logger.warning("LABVLA_SCHEMA_PATH entry %s is not a directory — skipping", d)
        return
    for py in sorted(d.glob("*.py")):
        if py.name.startswith("_"):
            continue
        # LABVLA_SCHEMA_PATH files are imported for their register() side
        # effects; a module that only register()s (no top-level SCHEMA) is
        # valid here, so allow it instead of raising.
        _load_file(py, allow_register_only=True)


def _load_file(
    path: Path,
    attr: Optional[str] = None,
    *,
    allow_register_only: bool = False,
) -> Optional[DatasetSchema]:
    """Import a .py file and return the named attr (default: `SCHEMA`).

    Files loaded this way may also call `register()` at import time;
    the registry catches any side effects.

    A module may legitimately expose no top-level ``SCHEMA`` (or set
    ``SCHEMA = None``) and instead ``register(...)`` one-or-more schemas by name
    at import time.
    When ``allow_register_only`` is True, such a module returns ``None`` instead
    of raising, so the caller can fall back to a registry lookup. An explicit
    ``attr`` (the user asked for ``:NAME``) that is present-but-wrong-type is
    still a hard error regardless.
    """
    spec = importlib.util.spec_from_file_location(f"_labvla_schema_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Python schema file {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # may call register() as side effect
    explicit_attr = attr is not None
    if attr is None:
        attr = "SCHEMA"
    if hasattr(mod, attr):
        val = getattr(mod, attr)
        if isinstance(val, DatasetSchema):
            return val
        # ``SCHEMA = None`` + register(...) is a valid register-only module.
        if val is None and not explicit_attr and allow_register_only:
            return None
        raise TypeError(
            f"{path}: attribute `{attr}` is {type(val).__name__}, expected DatasetSchema"
        )
    # No SCHEMA attr. If the file registered schemas by name instead, let the
    # caller retry a registry lookup; otherwise surface the missing attr.
    if allow_register_only and not explicit_attr:
        return None
    raise AttributeError(f"{path}: no top-level `{attr}` DatasetSchema found")


def resolve(spec: str) -> DatasetSchema:
    """Resolve `spec` to a DatasetSchema.

    Three forms (tried in order):
      1. Registered name     "labutopia_level3_transport_beaker"
      2. Dotted module[:ATTR] "my_pkg.my_schemas.foo"  or ":FOO"
      3. Absolute file[:ATTR] "/abs/path/schema.py:SCHEMA"

    Raises SchemaSpecError with a fuzzy suggestion on miss.
    """
    _ensure_autoload()

    # Form 3: absolute file path (possibly with :ATTR suffix).
    file_part, attr_part = _split_attr(spec)
    if file_part.endswith(".py") and Path(file_part).is_absolute():
        path = Path(file_part)
        if not path.exists():
            raise SchemaSpecError(f"schema file not found: {path}")
        # A register-only file (SCHEMA absent/None) imports its register()
        # side effects and returns None; fall through to the registry lookup /
        # miss-error below (which lists the names it registered) instead of
        # raising AttributeError.
        _pre = set(_REGISTRY)
        loaded = _load_file(path, attr_part, allow_register_only=True)
        if loaded is not None:
            return loaded
        # Register-only modules resolve by file path. Diff the registry around
        # the import: exactly one newly registered schema -> that's the answer;
        # several -> ambiguous, demand an explicit name.
        _new = [k for k in _REGISTRY if k not in _pre]
        if not _new:
            # In-repo modules are pre-registered by the autoload pass, so the
            # diff is empty — attribute by the registered schemas' source_path.
            _new = [k for k, v in _REGISTRY.items()
                    if getattr(v, "source_path", None)
                    and Path(v.source_path).resolve() == path.resolve()]
        if len(_new) == 1:
            return _REGISTRY[_new[0]]
        if len(_new) > 1:
            raise SchemaSpecError(
                f"{spec}: register-only file registered {len(_new)} schemas "
                f"({sorted(_new)}); pass the schema NAME instead of the file "
                f"path to disambiguate."
            )

    # Form 1: registry lookup.
    key = _ALIASES.get(spec, spec)
    if key in _REGISTRY:
        return _REGISTRY[key]

    # Form 2: dotted module path.
    if ":" in spec or "." in spec:
        mod_part, attr_part = _split_attr(spec)
        _pre_mod = set(_REGISTRY)
        try:
            mod = importlib.import_module(mod_part)
        except ModuleNotFoundError:
            pass
        else:
            attr_name = attr_part or "SCHEMA"
            explicit_attr = attr_part is not None
            if hasattr(mod, attr_name):
                val = getattr(mod, attr_name)
                if isinstance(val, DatasetSchema):
                    return val
                # ``SCHEMA = None`` (or a non-DatasetSchema SCHEMA) alongside
                # register(...) is a valid register-only module. When the user
                # did NOT ask for a specific ``:ATTR``, fall back to a registry
                # lookup (the import above ran register()) before treating the
                # wrong-type attr as a hard error.
                if not (val is None and not explicit_attr):
                    raise SchemaSpecError(
                        f"{spec}: `{attr_name}` is {type(val).__name__}, "
                        f"expected DatasetSchema"
                    )
            # Module imported but may have registered under a name.
            if spec in _REGISTRY:
                return _REGISTRY[spec]
            if spec in _ALIASES and _ALIASES[spec] in _REGISTRY:
                return _REGISTRY[_ALIASES[spec]]
            # Register-only module (SCHEMA=None + register() side effects) —
            # resolve via the registry diff like the file-path form. Exactly
            # one new schema -> return it; several -> demand an explicit name.
            _new_mod = [k for k in _REGISTRY if k not in _pre_mod]
            if not _new_mod and getattr(mod, "__file__", None):
                # In-repo modules are pre-registered by the autoload pass —
                # attribute by source_path instead of the (empty) diff.
                _mod_file = Path(mod.__file__).resolve()
                _new_mod = [k for k, v in _REGISTRY.items()
                            if getattr(v, "source_path", None)
                            and Path(v.source_path).resolve() == _mod_file]
            if len(_new_mod) == 1:
                return _REGISTRY[_new_mod[0]]
            if len(_new_mod) > 1:
                raise SchemaSpecError(
                    f"{spec}: register-only module registered {len(_new_mod)} "
                    f"schemas ({sorted(_new_mod)}); pass the schema NAME "
                    f"instead of the module path to disambiguate."
                )

    # Miss. Before reporting "unknown schema", surface any in-repo autoload
    # import failure — the spec may be missing precisely because the module
    # that would have registered it failed to import.
    _raise_if_autoload_failed()

    # Suggest closest names.
    candidates = registered_names()
    suggestions = difflib.get_close_matches(spec, candidates, n=3, cutoff=0.5)
    hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
    raise SchemaSpecError(
        f"unknown --dataset_schema {spec!r}.{hint} "
        f"Known names: {candidates}. "
        f"You can also pass a dotted module `pkg.mod:ATTR` or an absolute path "
        f"`/abs/path/file.py[:ATTR]`."
    )


def _split_attr(spec: str) -> tuple[str, Optional[str]]:
    """Split `'module:ATTR'` or `'/abs/path.py:ATTR'` into `(head, attr)`.

    Splits on the **rightmost** `:` so that absolute paths like
    `/a/b.py:SCHEMA` parse correctly. Only treats the right half as an attr
    when it's a valid Python identifier — otherwise returns `(spec, None)`.
    """
    if ":" not in spec:
        return spec, None
    idx = spec.rfind(":")
    right = spec[idx + 1:]
    if right.isidentifier():
        return spec[:idx], right
    return spec, None


__all__ = [
    "register",
    "resolve",
    "registered_names",
    "SchemaSpecError",
]
