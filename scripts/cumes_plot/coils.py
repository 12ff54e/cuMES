"""Coil-file readers and filament tube geometry."""

import json
import os

import numpy as np


def _circular_filament(radius, center_z, circular_points):
    phi = np.linspace(0.0, 2.0 * np.pi, circular_points, endpoint=False)
    return np.column_stack(
        [radius * np.cos(phi), radius * np.sin(phi),
         np.full_like(phi, center_z)])


def _load_json_coil_filaments(path, circular_points):
    try:
        with open(path, "r", encoding="utf-8") as stream:
            document = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"error: could not read coils file {path}: {exc}") \
            from exc

    if not isinstance(document, dict) or \
            document.get("schema") != "cumes-coils-v1":
        raise SystemExit(
            f"error: {path}: expected schema 'cumes-coils-v1'")
    periods = document.get("field_periods")
    if isinstance(periods, bool) or not isinstance(periods, int) or periods < 1:
        raise SystemExit(f"error: {path}: field_periods must be positive")
    circuits = document.get("circuits")
    if not isinstance(circuits, list) or not circuits:
        raise SystemExit(f"error: {path}: circuits must be a nonempty array")
    if any(not isinstance(circuit, dict) for circuit in circuits):
        raise SystemExit(f"error: {path}: each circuit must be an object")
    indices = [circuit.get("current_index") for circuit in circuits]
    if any(isinstance(index, bool) or not isinstance(index, int)
           for index in indices):
        raise SystemExit(
            f"error: {path}: every current_index must be an integer")
    circuits = sorted(circuits, key=lambda circuit: circuit.get(
        "current_index", -1))
    indices = [circuit.get("current_index") for circuit in circuits]
    if indices != list(range(len(circuits))):
        raise SystemExit(
            f"error: {path}: current_index values must be contiguous from zero")

    filaments = []
    for circuit in circuits:
        circuit_filaments = circuit.get("filaments")
        if not isinstance(circuit_filaments, list) or not circuit_filaments:
            raise SystemExit(
                f"error: {path}: every circuit needs a nonempty filaments array")
        for filament_data in circuit_filaments:
            if not isinstance(filament_data, dict):
                raise SystemExit(
                    f"error: {path}: each filament must be an object")
            winding = filament_data.get("winding_number")
            if isinstance(winding, bool) or not isinstance(winding, (int, float)) \
                    or not np.isfinite(winding) or winding == 0.0:
                raise SystemExit(
                    f"error: {path}: winding_number must be finite and nonzero")
            filament_type = filament_data.get("type")
            if filament_type == "polygon":
                try:
                    vertices = np.asarray(filament_data.get("vertices"),
                                          dtype=float)
                except (TypeError, ValueError) as exc:
                    raise SystemExit(
                        f"error: {path}: polygon vertices must be numeric") \
                        from exc
                if vertices.ndim != 2 or vertices.shape[0] < 3 or \
                        vertices.shape[1] != 3 or not np.isfinite(vertices).all():
                    raise SystemExit(
                        f"error: {path}: polygon vertices must be a finite Nx3 "
                        "array with N >= 3")
                if np.array_equal(vertices[0], vertices[-1]):
                    raise SystemExit(
                        f"error: {path}: JSON polygon closure is implicit; do "
                        "not repeat the first vertex")
                filaments.append(np.vstack([vertices, vertices[0]]))
            elif filament_type == "axisymmetric_circle":
                radius = filament_data.get("radius")
                center_z = filament_data.get("center_z")
                if isinstance(radius, bool) or \
                        not isinstance(radius, (int, float)) or \
                        not np.isfinite(radius) or radius <= 0.0:
                    raise SystemExit(
                        f"error: {path}: circle radius must be finite and positive")
                if isinstance(center_z, bool) or \
                        not isinstance(center_z, (int, float)) or \
                        not np.isfinite(center_z):
                    raise SystemExit(
                        f"error: {path}: circle center_z must be finite")
                filaments.append(_circular_filament(
                    radius, center_z, circular_points))
            else:
                raise SystemExit(
                    f"error: {path}: unsupported filament type {filament_type!r}")
    return periods, filaments


def load_coil_filaments(path, circular_points=160):
    """Read the polygon and axis-aligned circular filaments accepted by
    MAKEGRID coils-dot and cumes-coils-v1 JSON files.  Coordinates are
    Cartesian (x, y, z); circuit currents affect the field but not the
    geometry rendered here."""
    if os.path.splitext(path)[1].lower() == ".json":
        return _load_json_coil_filaments(path, circular_points)

    filaments = []
    vertices = []
    periods = None
    in_filaments = False
    found_end = False

    try:
        stream = open(path, "r", encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"error: could not open coils file {path}: {exc}") \
            from exc

    with stream:
        for line_number, raw_line in enumerate(stream, 1):
            line = raw_line.strip()
            if not line or line.startswith("!"):
                continue
            lower = line.lower()
            if not in_filaments:
                if lower.startswith("periods"):
                    fields = line.split()
                    if len(fields) != 2:
                        raise SystemExit(
                            f"error: {path}:{line_number}: expected "
                            "'periods <positive integer>'")
                    try:
                        periods = int(fields[1])
                    except ValueError as exc:
                        raise SystemExit(
                            f"error: {path}:{line_number}: invalid periods") \
                            from exc
                elif lower.startswith("begin filament"):
                    in_filaments = True
                continue

            if lower.startswith("mirror"):
                if not (lower.endswith("nil") or lower.endswith("nul")):
                    raise SystemExit(
                        f"error: {path}:{line_number}: coil mirroring is "
                        "unsupported; expected mirror NIL or NUL")
                continue
            if lower.startswith("end"):
                found_end = True
                break

            fields = line.split()
            if len(fields) not in (4, 6):
                raise SystemExit(
                    f"error: {path}:{line_number}: expected four or six "
                    "filament fields")
            try:
                vertex = tuple(float(value) for value in fields[:3])
            except ValueError as exc:
                raise SystemExit(
                    f"error: {path}:{line_number}: invalid filament vertex") \
                    from exc
            vertices.append(vertex)
            if len(fields) == 6:
                if len(vertices) == 1:
                    radius, _, center_z = vertices[0]
                    if radius <= 0.0:
                        raise SystemExit(
                            f"error: {path}:{line_number}: circular filament "
                            "radius must be positive")
                    filament = _circular_filament(
                        radius, center_z, circular_points)
                else:
                    filament = np.asarray(vertices, dtype=float)
                filaments.append(filament)
                vertices = []

    if not in_filaments:
        raise SystemExit(f"error: {path}: no 'begin filament' section")
    if not found_end:
        raise SystemExit(f"error: {path}: no terminating 'end' line")
    if vertices:
        raise SystemExit(f"error: {path}: file ends inside a filament")
    if periods is None or periods < 1:
        raise SystemExit(f"error: {path}: no positive periods value")
    if not filaments:
        raise SystemExit(f"error: {path}: no coil filaments")
    return periods, filaments


def resolve_coils_path(requested, params):
    """Resolve an explicit or recorded coil-configuration path. Relative
    paths are first interpreted exactly as the solver does (from the current
    working directory), then relative to the recorded input file when
    available."""
    path = requested or params.get("coils_file", "")
    if not path:
        raise SystemExit(
            "error: --coils needs PATH because this state contains a "
            "precomputed mgrid_file but no recorded coils_file path")
    candidates = [path]
    source_path = params.get("_source_path", "")
    if not os.path.isabs(path) and source_path:
        candidates.append(os.path.join(os.path.dirname(source_path), path))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    tried = ", ".join(os.path.abspath(candidate) for candidate in candidates)
    raise SystemExit(f"error: coils file not found; tried {tried}")


def tube_mesh(points, radius, n_sides=10):
    """Build a closed tube around a filament centerline using a smoothly
    transported local frame.  Coils-dot polygon filaments are closed by
    convention; explicitly repeated terminal vertices are collapsed before
    the periodic mesh is formed."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("filament points must have shape (n, 3)")
    keep = np.ones(len(points), dtype=bool)
    if len(points) > 1:
        keep[1:] = np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-12
    points = points[keep]
    if len(points) > 2 and np.linalg.norm(points[-1] - points[0]) <= 1e-12:
        points = points[:-1]
    if len(points) < 3:
        raise ValueError("a tube filament needs at least three unique points")

    tangent = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    tangent_norm = np.linalg.norm(tangent, axis=1)
    if np.any(tangent_norm <= 1e-14):
        raise ValueError("filament has a degenerate tangent")
    tangent /= tangent_norm[:, None]

    axes = np.eye(3)
    reference = axes[np.argmin(np.abs(axes @ tangent[0]))]
    normal = np.empty_like(points)
    normal[0] = np.cross(tangent[0], reference)
    normal[0] /= np.linalg.norm(normal[0])
    for i in range(1, len(points)):
        transported = normal[i - 1] - np.dot(normal[i - 1], tangent[i]) * \
            tangent[i]
        magnitude = np.linalg.norm(transported)
        if magnitude <= 1e-12:
            reference = axes[np.argmin(np.abs(axes @ tangent[i]))]
            transported = np.cross(tangent[i], reference)
            magnitude = np.linalg.norm(transported)
        normal[i] = transported / magnitude
    binormal = np.cross(tangent, normal)

    points = np.vstack([points, points[0]])
    normal = np.vstack([normal, normal[0]])
    binormal = np.vstack([binormal, binormal[0]])
    angle = np.linspace(0.0, 2.0 * np.pi, n_sides + 1)
    mesh = points[:, :, None] + radius * (
        normal[:, :, None] * np.cos(angle)[None, None, :] +
        binormal[:, :, None] * np.sin(angle)[None, None, :])
    return mesh[:, 0, :], mesh[:, 1, :], mesh[:, 2, :]
