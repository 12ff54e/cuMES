"""Output-name handling shared by the plotting backends."""

import os


DEFAULT_OUTPUT_PATH = "figure_data/equilibrium.png"


def resolve_output_base(output_path=None, output_directory=None):
    """Return a prefixed base path or a directory-only base path."""
    if output_path is not None and output_directory is not None:
        raise ValueError("output path and output directory are mutually exclusive")
    if output_directory is not None:
        if not output_directory:
            raise ValueError("output directory must not be empty")
        return os.path.join(output_directory, "")
    return os.path.splitext(output_path or DEFAULT_OUTPUT_PATH)[0]


def figure_path(base, suffix):
    """Build one PNG name, omitting the separator prefix in directory mode."""
    separators = (os.sep,) if os.altsep is None else (os.sep, os.altsep)
    if base.endswith(separators):
        return os.path.join(base, f"{suffix}.png")
    return f"{base}_{suffix}.png"
