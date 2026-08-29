"""Read Boozer-v2 containers and construct magnetic-coordinate plot grids.

The public Boozer format stores geometry as six real Fourier families on the
mixed ``(s, theta_b, zeta)`` grid.  The source toroidal angle is unchanged and

    theta_b = theta_p + iota * nu.

The stored spectra are sufficient for the Boozer plots. Exact PEST plots use
the native cuMES state instead: reconstructing them from these already
transformed and truncated spectra would only produce an approximation.
"""

from dataclasses import dataclass
import struct

import numpy as np


MAGIC = b"MCBOOZ02"
VERSION = 2
PERIOD = 2.0 * np.pi
FAMILY_NAMES = (
    "rmncc", "rmnss", "zmnsc", "zmncs", "numnsc", "numncs",
)
SCHEMA = "magnetic-coordinate-boozer-v2"
COORDINATE_CONVENTION = (
    "mixed-grid-v1: theta_b uniform; zeta is the unchanged source toroidal "
    "angle; zeta_b=zeta+nu"
)
FOURIER_CONVENTION = (
    "real-parity-v2: f=sum[cc*cos(m*theta_b)*cos(n*zeta) + "
    "ss*sin(m*theta_b)*sin(n*zeta)] for even fields and "
    "f=sum[sc*sin(m*theta_b)*cos(n*zeta) + "
    "cs*cos(m*theta_b)*sin(n*zeta)] for odd fields; m,n are nonnegative "
    "and n is a field-period mode"
)


@dataclass(frozen=True)
class BoozerData:
    """Validated, backend-independent Boozer-v2 data."""

    source_path: str
    source_ns: int
    nfp: int
    first_surface: int
    ntheta: int
    nzeta: int
    s: np.ndarray
    iota: np.ndarray
    mode_m: np.ndarray
    mode_n: np.ndarray
    families: dict
    b: np.ndarray

    @property
    def surface_count(self):
        return self.source_ns - self.first_surface

    @property
    def theta_b(self):
        return PERIOD * np.arange(self.ntheta) / self.ntheta

    @property
    def zeta(self):
        return PERIOD * np.arange(self.nzeta) / self.nzeta


@dataclass(frozen=True)
class CoordinateGrid:
    """Geometry and |B| on a regular magnetic-coordinate angular grid."""

    coordinate: str
    angle_label: str
    s: np.ndarray
    theta: np.ndarray
    zeta: np.ndarray
    r: np.ndarray
    z: np.ndarray
    b: np.ndarray


class _BinaryReader:
    def __init__(self, path):
        try:
            self._stream = open(path, "rb")
        except OSError as exc:
            raise SystemExit(f"error: could not read Boozer file {path}: {exc}") \
                from exc
        self.path = path

    def close(self):
        self._stream.close()

    def read(self, size):
        value = self._stream.read(size)
        if len(value) != size:
            raise SystemExit(f"error: truncated Boozer file {self.path}")
        return value

    def i32(self):
        return struct.unpack("<i", self.read(4))[0]

    def f64(self):
        return struct.unpack("<d", self.read(8))[0]

    def string(self):
        size = self.i32()
        if not 0 <= size < 1 << 24:
            raise SystemExit(
                f"error: corrupt string length in Boozer file {self.path}")
        try:
            return self.read(size).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SystemExit(
                f"error: invalid UTF-8 in Boozer file {self.path}") from exc

    def array(self, dtype, count):
        itemsize = np.dtype(dtype).itemsize
        return np.frombuffer(self.read(itemsize * count), dtype=dtype).copy()


def _load_binary(path):
    reader = _BinaryReader(path)
    try:
        if reader.read(8) != MAGIC or reader.i32() != VERSION:
            raise SystemExit(f"error: unsupported Boozer binary format in {path}")
        coordinate_convention = reader.string()
        fourier_convention = reader.string()
        if coordinate_convention != COORDINATE_CONVENTION or \
                fourier_convention != FOURIER_CONVENTION:
            raise SystemExit(
                f"error: unsupported Boozer coordinate convention in {path}")
        source_path = reader.string()
        header = [reader.i32() for _ in range(13)]
        (source_format_version, source_ns, source_ntheta, source_nzeta,
         source_mpol, source_ntor, nfp, first_surface, ntheta, nzeta,
         mmax, nmax, radial_order) = header
        resonance_tolerance = reader.f64()
        del source_format_version, source_ntheta, source_nzeta, source_mpol
        del source_ntor, radial_order, resonance_tolerance
        if mmax < 0 or nmax < 0:
            raise SystemExit(f"error: invalid Boozer mode bounds in {path}")
        surfaces = source_ns - first_surface
        modes = (mmax + 1) * (nmax + 1)
        if not (1 <= surfaces <= 1 << 20 and 1 <= modes <= 1 << 20 and
                ntheta >= 1 and nzeta >= 1):
            raise SystemExit(f"error: invalid Boozer dimensions in {path}")
        s = reader.array("<f8", surfaces)
        iota = reader.array("<f8", surfaces)
        mode_m = reader.array("<i4", modes)
        mode_n = reader.array("<i4", modes)
        families = {
            name: reader.array("<f8", surfaces * modes).reshape(surfaces, modes)
            for name in FAMILY_NAMES
        }
        b = reader.array("<f8", surfaces * nzeta * ntheta).reshape(
            surfaces, nzeta, ntheta)
        reader.array("<f8", surfaces * nzeta * ntheta)  # sqrt_g_b
        reader.array("<f8", surfaces)  # b2j00
    finally:
        reader.close()
    return BoozerData(source_path, source_ns, nfp, first_surface, ntheta,
                      nzeta, s, iota, mode_m, mode_n, families, b)


def _decode(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _load_netcdf(path):
    try:
        from scipy.io import netcdf_file
        with netcdf_file(path, "r", mmap=False) as nc:
            if _decode(getattr(nc, "schema", "")) != SCHEMA:
                raise SystemExit(f"error: unsupported Boozer NetCDF schema in {path}")

            def attr(name):
                return int(getattr(nc, name))

            source_path = _decode(getattr(nc, "source_path", ""))
            source_ns = attr("source_ns")
            nfp = attr("nfp")
            first_surface = attr("first_surface")
            ntheta = attr("ntheta")
            nzeta = attr("nzeta")
            arrays = {name: np.asarray(nc.variables[name][:]).copy()
                      for name in ("s", "iota", "mode_m", "mode_n", "B")}
            families = {name: np.asarray(nc.variables[name][:], dtype=float).copy()
                        for name in FAMILY_NAMES}
    except OSError as exc:
        raise SystemExit(f"error: could not read Boozer file {path}: {exc}") \
            from exc
    return BoozerData(source_path, source_ns, nfp, first_surface, ntheta,
                      nzeta, np.asarray(arrays["s"], dtype=float),
                      np.asarray(arrays["iota"], dtype=float),
                      np.asarray(arrays["mode_m"], dtype=np.int32),
                      np.asarray(arrays["mode_n"], dtype=np.int32), families,
                      np.asarray(arrays["B"], dtype=float))


def _load_hdf5(path):
    try:
        import h5py
        with h5py.File(path, "r") as h5:
            if _decode(h5.attrs.get("schema", "")) != SCHEMA:
                raise SystemExit(f"error: unsupported Boozer HDF5 schema in {path}")
            source_path = _decode(h5.attrs.get("source_path", ""))
            source_ns = int(h5.attrs["source_ns"])
            nfp = int(h5.attrs["nfp"])
            first_surface = int(h5.attrs["first_surface"])
            ntheta = int(h5.attrs["ntheta"])
            nzeta = int(h5.attrs["nzeta"])
            s = np.asarray(h5["s"][:], dtype=float)
            iota = np.asarray(h5["iota"][:], dtype=float)
            mode_m = np.asarray(h5["mode_m"][:], dtype=np.int32)
            mode_n = np.asarray(h5["mode_n"][:], dtype=np.int32)
            families = {name: np.asarray(h5[name][:], dtype=float)
                        for name in FAMILY_NAMES}
            b = np.asarray(h5["B"][:], dtype=float)
    except OSError as exc:
        raise SystemExit(f"error: could not read Boozer file {path}: {exc}") \
            from exc
    return BoozerData(source_path, source_ns, nfp, first_surface, ntheta,
                      nzeta, s, iota, mode_m, mode_n, families, b)


def _validate(data, path):
    surfaces = data.source_ns - data.first_surface
    modes = data.mode_m.size
    if data.nfp < 1 or data.first_surface < 1 or surfaces < 1 or \
            data.ntheta < 4 or data.nzeta < 1:
        raise SystemExit(f"error: invalid Boozer dimensions in {path}")
    if data.s.shape != (surfaces,) or data.iota.shape != (surfaces,) or \
            data.mode_n.shape != (modes,) or data.b.shape != (
                surfaces, data.nzeta, data.ntheta):
        raise SystemExit(f"error: inconsistent Boozer array extents in {path}")
    if modes < 1 or any(np.asarray(data.families[name]).shape != (surfaces, modes)
                        for name in FAMILY_NAMES):
        raise SystemExit(f"error: inconsistent Boozer spectrum extents in {path}")
    real_arrays = [data.s, data.iota, data.b]
    real_arrays.extend(data.families.values())
    if any(not np.all(np.isfinite(array)) for array in real_arrays):
        raise SystemExit(f"error: non-finite data in Boozer file {path}")
    if np.any(data.b <= 0.0) or np.any(np.diff(data.s) <= 0.0):
        raise SystemExit(f"error: nonphysical Boozer grid in {path}")
    if np.any(data.mode_m < 0) or np.any(data.mode_n < 0):
        raise SystemExit(f"error: negative Boozer mode number in {path}")
    return data


def load_boozer(path):
    """Read a binary, NetCDF, or HDF5 Boozer-v2 result container."""
    try:
        with open(path, "rb") as stream:
            magic = stream.read(8)
    except OSError as exc:
        raise SystemExit(f"error: could not read Boozer file {path}: {exc}") \
            from exc
    if magic == MAGIC:
        data = _load_binary(path)
    elif magic.startswith(b"CDF"):
        data = _load_netcdf(path)
    elif magic.startswith(b"\x89HDF"):
        data = _load_hdf5(path)
    else:
        raise SystemExit(f"error: unrecognized Boozer container {path}")
    return _validate(data, path)


def _synthesize_boozer_geometry(data, theta, zeta):
    """Synthesize R and Z on a requested uniform mixed grid."""
    mt = data.mode_m[:, None] * theta[None, :]
    nz = data.mode_n[:, None] * zeta[None, :]
    cos_mt, sin_mt = np.cos(mt), np.sin(mt)
    cos_nz, sin_nz = np.cos(nz), np.sin(nz)

    def combine(first, first_t, first_z, second, second_t, second_z):
        return (np.einsum("sm,mz,mt->szt", data.families[first], first_z,
                          first_t, optimize=True) +
                np.einsum("sm,mz,mt->szt", data.families[second], second_z,
                          second_t, optimize=True))

    r = combine("rmncc", cos_mt, cos_nz, "rmnss", sin_mt, sin_nz)
    z = combine("zmnsc", sin_mt, cos_nz, "zmncs", cos_mt, sin_nz)
    return r, z


def _resample_uniform_axis(values, target_count, axis):
    """Linearly resample one uniformly periodic array axis."""
    source_count = values.shape[axis]
    if source_count == target_count:
        return values.copy()
    position = np.arange(target_count) * source_count / target_count
    lower = np.floor(position).astype(int) % source_count
    upper = (lower + 1) % source_count
    fraction = position - np.floor(position)
    shape = [1] * values.ndim
    shape[axis] = target_count
    fraction = fraction.reshape(shape)
    return ((1.0 - fraction) * np.take(values, lower, axis=axis) +
            fraction * np.take(values, upper, axis=axis))


def _resample_field(data, ntheta, nzeta):
    field = _resample_uniform_axis(data.b, ntheta, axis=-1)
    return _resample_uniform_axis(field, nzeta, axis=-2)


def _periodic_interp(target, source, values):
    source = np.mod(source, PERIOD)
    order = np.argsort(source)
    source = source[order]
    values = np.asarray(values)[order]
    if np.min(np.diff(source)) <= 64.0 * np.finfo(float).eps:
        raise ValueError("magnetic poloidal map contains duplicate angles")
    return np.interp(target, source, values, period=PERIOD)


def make_boozer_grid(data, ntheta=None, nzeta=None):
    """Return geometry and |B| on the regular mixed Boozer grid."""
    ntheta = data.ntheta if ntheta is None else int(ntheta)
    nzeta = data.nzeta if nzeta is None else int(nzeta)
    if ntheta < 4 or nzeta < 1:
        raise ValueError("coordinate-grid angular dimensions are invalid")
    theta = PERIOD * np.arange(ntheta) / ntheta
    zeta = PERIOD * np.arange(nzeta) / nzeta
    r, z = _synthesize_boozer_geometry(data, theta, zeta)
    b = _resample_field(data, ntheta, nzeta)
    return CoordinateGrid("Boozer", r"$\theta_b$", data.s.copy(),
                          theta, zeta, r, z, b)


def make_pest_grid(s, theta, zeta, r, z, b, lambda_physical):
    """Remap native-state fields from theta onto a regular PEST grid.

    ``lambda_physical`` is the normalized displacement reconstructed from the
    native cuMES lambda coefficients, and ``theta_p = theta + lambda``. Every
    real-space input has shape ``[nonaxis_surface, zeta, theta]``.
    """
    s = np.asarray(s, dtype=float)
    theta = np.asarray(theta, dtype=float)
    zeta = np.asarray(zeta, dtype=float)
    arrays = [np.asarray(value, dtype=float)
              for value in (r, z, b, lambda_physical)]
    r, z, b, lambda_physical = arrays
    expected = (s.size, zeta.size, theta.size)
    if theta.ndim != 1 or zeta.ndim != 1 or s.ndim != 1 or \
            theta.size < 4 or zeta.size < 1 or any(
                value.shape != expected for value in arrays):
        raise ValueError("native PEST plotting arrays have inconsistent extents")
    if any(not np.all(np.isfinite(value))
           for value in [s, theta, zeta] + arrays):
        raise ValueError("native PEST plotting arrays contain non-finite data")
    if np.any(b <= 0.0) or np.any(np.diff(s) <= 0.0):
        raise ValueError("native PEST plotting grid is nonphysical")

    r_p = np.empty_like(r)
    z_p = np.empty_like(z)
    b_p = np.empty_like(b)
    target = theta
    for surface in range(s.size):
        for toroidal in range(zeta.size):
            theta_p = target + lambda_physical[surface, toroidal]
            closed = np.unwrap(np.append(theta_p, theta_p[0] + PERIOD))
            if np.any(np.diff(closed) <= 0.0):
                raise ValueError(
                    "native theta-to-PEST map is not orientation preserving")
            r_p[surface, toroidal] = _periodic_interp(
                target, theta_p, r[surface, toroidal])
            z_p[surface, toroidal] = _periodic_interp(
                target, theta_p, z[surface, toroidal])
            b_p[surface, toroidal] = _periodic_interp(
                target, theta_p, b[surface, toroidal])
    return CoordinateGrid("PEST", r"$\theta_p$", s.copy(), target,
                          zeta, r_p, z_p, b_p)


def interpolate_zeta(values, zeta):
    """Periodically interpolate an ``[..., zeta, theta]`` array."""
    nzeta = values.shape[-2]
    grid_index = (zeta % PERIOD) * nzeta / PERIOD
    lower = int(np.floor(grid_index)) % nzeta
    fraction = grid_index - np.floor(grid_index)
    upper = (lower + 1) % nzeta
    return (1.0 - fraction) * values[..., lower, :] + \
        fraction * values[..., upper, :]
