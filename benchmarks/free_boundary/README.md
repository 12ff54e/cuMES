# Free-boundary performance matrix

The matrix retains Solovev, CTH-like and W7-X source workloads with every
configured stage, tolerance and iteration cap. Solovev's two field-source forms
cover precomputed-table loading and embedded MAKEGRID setup; they share the
same physics. W7-X already has one stage in its upstream input.

| Case | Radial stages | Tolerances | Iteration caps | Full-vacuum spacing |
| --- | --- | --- | --- | ---: |
| `solovev_mgrid` | 16, 32 | 1e-10, 1e-14 | 10000, 20000 | 6 |
| `solovev_embedded` | 16, 32 | 1e-10, 1e-14 | 10000, 20000 | 6 |
| `cth_multigrid` | 15, 25 | 1e-8, 1e-10 | 2500, 2500 | 9 |
| `w7x_original` | 51 | 1e-12 | 50000 | 6 |
| `w7x_positive_flux` | 51 | 1e-12 | 50000 | 6 |

The first four cases were selected from source/data inspection before timing.
The fifth was separately declared before its own runs to reproduce the earlier
cuMES W7-X convention. It changes only `phiedge` from `-1.74` to `+1.74` and is
always reported separately; the original negative-flux input and its failures
remain in the matrix. Initial runs used immutable materialized inputs before
this manifest was committed. Their hashes are recorded in the run protocols;
the manifest does not claim to predate them.

`sources/` retains the byte-for-byte input sources. The manifest records source
revisions and hashes, every adaptation, all stage controls, and field-data hashes.
CTH and W7-X inputs come from VMEC++ v0.7.0 commit
`335ef66441d82980331d0062ab8d0398eff50818`, under the accompanying MIT license.
Solovev inputs come from cuMES commit
`bcdd3dab5149dc0c5ec8608574f59f2d6932b398`. Apart from the explicit positive-flux
supplement, only field paths and W7-X's ignored `free_boundary_method=only_coils`
selector change. No convergence controls or other physical values change.

Large mgrid tables remain outside Git. The original local experiment stores
them in `../tmp/cumes-free-boundary-20260909/data/`. Solovev and CTH tables match
their upstream v0.7.0 Git LFS object hashes; a 133-byte Git LFS pointer is not a
field table. W7-X uses the existing 88,550,256-byte table from
`../cuMES/figure_data/w7x_free_boundary/mgrid_w7x.nc`, pinned by content hash.
Its exact generating binary revision is unavailable; the original generating
parameter file and coil-file hash are retained for provenance. The compact
Solovev coil file is tracked in `sources/` with its upstream license.

Regenerate or check the portable fixtures without running a solver:

```bash
python3 benchmarks/free_boundary/generate.py
python3 benchmarks/free_boundary/generate.py --check
```

Template field paths are portable data filenames. Materialize a fresh directory
before running, pointing `--data-root` to the three pinned mgrid tables:

```bash
python3 benchmarks/free_boundary/generate.py \
  --materialize /tmp/cumes-free-boundary-inputs \
  --data-root ../tmp/cumes-free-boundary-20260909/data
```

This checks every dependency's size/hash, resolves field paths absolutely and
writes a manifest with the resulting input hashes. The Solovev coil file may
reside in the data root or fall back to its tracked copy. The materializer
refuses to overwrite a nonempty directory. Copy the fixture directory and
pinned field data to another machine, then materialize there; no input physics
depends on the machine's filesystem layout.

The root study initially used the already materialized inputs under
`../tmp/cumes-free-boundary-20260909/inputs/`; their path strings and hashes are
preserved in the run protocols. Their values match these templates after
resolving the declared dependency keys. Regeneration does not overwrite them.

`run.py` accepts a materialized manifest or input directory and runs complete
baseline/candidate pairs. `analyze.py` checks the native reports and numerical
results. See each script's `--help` for arguments. Keep failed runs, all timings
and regressions; do not adjust the matrix after outcomes. Run only one benchmark
on a given GPU at a time. Process wall time includes field setup and output;
the reported solver interval includes vacuum-coupled iteration work but excludes
outer field setup and publication output.
