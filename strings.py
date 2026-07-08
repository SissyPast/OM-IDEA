"""
Sphinx extension to display attrs field metadata in documentation.

Add to your conf.py:
    extensions = ['sphinx_attrs_metadata']

Or if the file is in your docs directory:
    import sys
    sys.path.insert(0, os.path.abspath('.'))
    extensions = ['sphinx_attrs_metadata']
"""

import inspect
from typing import Any, List, Optional

import attrs


def _make_crossref(obj: Any) -> Optional[str]:
    """Create a Sphinx cross-reference for an object (function, class, etc.)."""
    if obj is None:
        return None

    # Get the module and qualified name
    module = getattr(obj, "__module__", None)
    name = getattr(obj, "__qualname__", None) or getattr(obj, "__name__", None)

    if not name:
        return None

    # Build the full reference path
    if module and not module.startswith("builtins"):
        full_path = f"{module}.{name}"
    else:
        full_path = name

    # Determine the appropriate role based on object type
    if inspect.isclass(obj):
        role = "class"
    elif inspect.isfunction(obj) or inspect.ismethod(obj):
        role = "func"
    else:
        role = "obj"

    # Use :py: domain explicitly and ~ prefix to show only the short name
    return f":py:{role}:`~{full_path}`"


def get_attrs_metadata_docs(cls: type) -> Optional[List[str]]:
    """Extract metadata documentation from attrs fields."""
    if not attrs.has(cls):
        return None

    fields = attrs.fields(cls)
    if not fields:
        return None

    # Only include fields that have non-empty metadata
    fields_with_metadata = [(f.name, f.type, f.metadata) for f in fields if f.metadata]
    if not fields_with_metadata:
        return None

    lines = []
    lines.append("")
    lines.append(".. rubric:: Field Metadata")
    lines.append("")

    for name, type_ann, metadata in fields_with_metadata:
        # Field name with optional type
        if type_ann is not None:
            # type_str = _format_type(type_ann)
            lines.append(f"- **{name}** ({_make_crossref(type_ann)})")
        else:
            lines.append(f"- **{name}**")

        # Display each metadata key-value pair
        for key, value in metadata.items():
            value_str = _format_value(value)
            lines.append(f"   - ``{key}``: ``{value_str}``")

        lines.append("")

    return lines


def _format_type(type_annotation: Any) -> str:
    """Format a type annotation for display."""
    if type_annotation is None:
        return "Any"

    if hasattr(type_annotation, "__name__"):
        return type_annotation.__name__

    type_str = str(type_annotation)
    for prefix in ("typing.", "typing_extensions."):
        type_str = type_str.replace(prefix, "")

    return type_str


def _format_value(value: Any) -> str:
    """Format a metadata value for display."""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return repr(value)
    if isinstance(value, (list, tuple, set)):
        if len(value) == 0:
            return repr(value)
        if len(value) <= 5:
            return repr(value)
        return f"{type(value).__name__} with {len(value)} items"
    if isinstance(value, dict):
        if len(value) == 0:
            return "{}"
        if len(value) <= 3:
            return repr(value)
        return f"dict with {len(value)} keys"
    if callable(value):
        return getattr(value, "__name__", repr(value))

    value_repr = repr(value)
    if len(value_repr) > 60:
        return value_repr[:57] + "..."
    return value_repr


def process_docstring(
    app,
    what: str,
    name: str,
    obj: Any,
    options,
    lines: List[str],
) -> None:
    """Autodoc event handler to append attrs metadata documentation."""
    if what != "class":
        return

    try:
        if not attrs.has(obj):
            return
    except TypeError:
        return

    metadata_docs = get_attrs_metadata_docs(obj)
    if metadata_docs:
        lines.extend(metadata_docs)


def setup(app):
    """Sphinx extension setup."""
    app.connect("autodoc-process-docstring", process_docstring)

    return {
        "version": "0.1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
