# TLA Tests (3-in-1, by op category)

Unified test scripts for TLA ops. Each script runs three stages in one file,
driven by env `TLA_TEST_STAGE`:

| Stage     | Verifies                                         | NPU needed |
|-----------|--------------------------------------------------|-----------|
| `ir`      | offline HIVM IR handed to npuir (structure asserts) | no      |
| `compile` | `.o`/`.npubin` artifacts in `TRITON_CACHE_DIR`      | yes     |
| `run`     | device launch + accuracy (`CAST_BACK OK`, ...)      | yes     |

Default `all` auto-degrades: ir only when no NPU, full chain when NPU is up.
Triton references live in the same case file (`TRITON_CAST_BACK OK`, ...).

## Layout

- `vector/` — SIMD/AIV ops (current 9 cases).
- `utils/` — shared helpers (`tla_test_utils.py`).
- `cube/` `mix/` — future matmul / mixed ops, same template.
- `simd_frontend_contract.py` — host-only contract tests for the elementwise
  SIMD ops (no BuildIR build, no NPU: it loads `language/tla/simd.py` from the
  source tree and drives it with a recording mock builder).
- `simd_api_validation/` — device-agnostic validation harness for the same
  elementwise APIs: case registry, deterministic inputs, CPU reference, cross-side
  comparator and host tests behind `--list` / `--check` / `--case`.  Its
  `conftest.py` keeps the harness out of the `*.py`-wide collection above;
  only its `tests/test_harness.py` (host tests) is collected.  See
  `simd_api_validation/README.md`.

## Elementwise SIMD ops (`tla.simd.vdiv/vmax/vmin/vmaxs/vexp/vdup/vdups/vdupv`)

Support table (derived from the AVE lowering, not from the TileLang names):

| API      | AVE op | element types | notes |
|----------|--------|---------------|-------|
| `vdiv`   | `ave.hir.vdiv` (+`cast=cast_unsigned`) | f16/f32/i16/i32/i64 (+u16/u32/u64) | i8/u8/bf16 rejected; for `MODE_ZEROING`, a caller-supplied mask on unsigned 16/32-bit division appends an explicit `vsel(result, 0)` because that software sequence does not zero inactive lanes by itself (`MODE_MERGING` reuses its builder-side zero merge instead) |
| `vmax`   | `ave.hir.vmax` / `ave.hir.vumax` | f16/bf16/f32/i8/i16/i32/i64 (+u8..u64) | unsigned picks the U op |
| `vmin`   | `ave.hir.vmin` / `ave.hir.vumin` | same as `vmax` | |
| `vmaxs`  | `ave.hir.vmaxs` / `ave.hir.vumaxs` | f16/f32/i8/i16/i32/i64 (+u8..u64) | scalar must match the element type; i64/u64 constexpr scalars are exact (u64 is passed as two's-complement i64 bits, so `[2**63, 2**64)` works); out-of-range values raise `ValueError` |
| `vexp`   | `ave.hir.vexp` | f16/f32 | VFExpOp has no other lowering |
| `vdup`   | `ave.hir.broadcast` (masked `vdups`) | i8/i16/i32/bf16/f16/f32/fp8e4nv (u8/u16/u32) | i64/u64 rejected: the 64-bit library broadcast drops the predicate (an explicit mask would be silently ignored); use `vsplat` for an unmasked i64/u64 broadcast. Lanes = 2048/bits for the supported widths |
| `vdups`  | `ave.hir.broadcast` (masked `vdups`) | same as `vdup` | source-compatible alias of `vdup`; the AVE/CCE intrinsic name is `vdups` |
| `vdupv`  | `ave.hir.vector_broadcast` | same as `vdup` plus i64/u64 | `pos=POS.LOWEST` duplicates lane 0 (`vdup.z`), `pos=POS.HIGHEST` duplicates the last lane (`vdupm.z`); single-lane broadcast, not a half-register copy. The 64-bit form is a real 64-lane `_mlir_ciface_vdup_int64_t` -> `ave_vdup_i64` wrapper that forwards the predicate/position, so i64/u64 use `vector<64xi64>` + `vector<64xi1>` (unlike masked scalar `vdups`, whose 64-bit wrapper drops the predicate) |

Mode / precision:

- `mode`: both `tla.simd.MODE.ZEROING` and `tla.simd.MODE.MERGING` are
  accepted and forwarded to the TLA C++ builder, which consumes the mode for
  real (`tla_builder.cc::applySimdMode`). `ZEROING` emits the bare masked AVE
  op; `MERGING` emits the same op plus an `ave.hir.vsel` that merges a
  same-typed `+0` SSA vector (`arith.constant` zero +
  `ave.hir.scalar_broadcast`) into the inactive lanes. The merge source
  therefore defaults to `+0`, so both modes read inactive lanes as zero, but
  `MODE.MERGING` is **not** CANN's native dst-preserving predicate and does
  **not** reproduce TileLang's dst/src0-preserving merge. Use
  `vsel(mask, op(...), old)` when inactive lanes must keep the old value.
  Unknown modes still raise `ValueError`.  IR/device checks of the MERGING
  lowering need a TLA extension rebuilt from this tree; a stale `.so` still
  emits the old bare-op MERGING IR.
- `precision`: `None` / `'ftz_true'` keep the hardware SFU instruction.
  `precision=None` here is **hardware**, unlike TileLang where it follows
  `enable_fast_math`. For **f32 vdiv**, `'exact'` / `'vdiv_0ulp_ftz_true'` map
  onto the real high-precision AVE op `ave.hir.vdivfhp` (VFDivFHPOp), whose
  pinned regbase lowering is TileLang's 0ULP algorithm; the caller mask is
  forwarded. It is f32-only (f16 raises `ValueError`), matching TileLang.
  `vexp`'s `'ftz_false'` / `'vexp_1ulp_ftz_false'` still has no committed AVE
  lowering and raises `NotImplementedError` — never silently degraded.

`vmaxs` takes a constexpr number or a same-typed scalar tensor.  The constexpr
paths (`i64` above 2**53 and the whole unsigned `u64` range) are materialized
as exact integer constants; the scalar-tensor path carries the value at
runtime and is covered by `simd_elementwise_dtype_ir`.

`simd_tail_mask_ir` covers the dynamic long/tail shape: a runtime `N` is
broadcast with `vdup`, a runtime `vci` lane index is compared with one signed
`vcmp <LT>`, and a 5-chunk / step-2 loop feeds nine ALL-store arithmetic rows
plus four real masked-store rows.  It asserts that `pge` is present only as the
ALL mask and never as a 31/63/... tail prefix.

### Running against the in-tree build

The host contract tests load `language/tla/simd.py` from the source tree and
mock the builder, so they only need `triton` importable:

```bash
# host-only contract tests (no NPU, no build; mock builder)
TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  pytest third_party/ascend/unittest/tla_ut -k simd_frontend
```

The IR/compile/run stages import the **TLA Python package plus the compiled
bindings**, so use a build tree or an installed wheel built from a source tree
that contains these APIs.  A build that predates them shows up as a missing
`tla.simd.vdup` or `tla.view expects a tla.buffer` failure.  From the
repository root, with `PYTHONPATH` pointing at such a build tree (or with the
matching wheel installed):

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

TLA_TEST_STAGE=ir \
  pytest third_party/ascend/unittest/tla_ut/vector -k simd_elementwise
TLA_TEST_STAGE=ir \
  pytest third_party/ascend/unittest/tla_ut        # whole tla_ut, IR only
```

C++ changes require a normal rebuild.  For **Python-only** changes (including
`simd.py`), no C++ rebuild is needed, but the interpreter must import the
updated `language/tla` sources: use a matching build tree/wheel, or an
editable install of the source tree.  The `ir` stage stops at the HIVM/AVE IR
handed to npuir; it does not exercise bishengir lowering or a device, so dtype
support must also be checked against the lowering sources / compile+run
stages.

## How to verify

```bash
# same PYTHONPATH/environment as above
cd third_party/ascend/unittest/tla_ut/vector
TLA_TEST_STAGE=ir      python cast_back.py   # 1. IR only, no NPU
TLA_TEST_STAGE=compile python cast_back.py   # 2. artifacts (needs Ascend driver)
TLA_TEST_STAGE=run     python cast_back.py   # 3. real-device accuracy (needs NPU)
python cast_back.py                          # 4. all (auto-degrade)
```

On a real NPU machine: use an environment that provides torch, torch_npu and a
TLA-capable Triton build, set `ASCEND_RT_VISIBLE_DEVICES` and
`TRITON_CACHE_DIR` for that installation, and run each case. Pass criteria:
`IR PASS` / `COMPILE PASS` / `<OP> OK` / `ALL PASS: <op> ir compile run`.

## One-click pytest runner

The same cases are pytest-collectable (`conftest.py` collects every `*.py` via
`pytest_collect_file` since cases are named by op, not `test_*`; it also maps
`TLA_TEST_STAGE` to skips and provides the `output_dir` fixture), so any
directory level can be run directly with pytest.  The IR/compile/run variants
need the build `PYTHONPATH` shown above:

```bash
TLA_TEST_STAGE=ir pytest third_party/ascend/unittest/tla_ut/vector   # IR only, no NPU
TLA_TEST_STAGE=compile pytest third_party/ascend/unittest/tla_ut/vector
TLA_TEST_STAGE=run pytest third_party/ascend/unittest/tla_ut/vector
pytest third_party/ascend/unittest/tla_ut/vector                     # all (auto-degrade)
pytest third_party/ascend/unittest/tla_ut                            # whole tla_ut
pytest .../tla_ut/vector -k rope                                      # subset
```

Collection and stage logic live in `tla_ut/conftest.py` and apply to the whole
`tla_ut` subtree from any working directory (including `pytest .` at the
`unittest` root). The root `unittest/conftest.py` NPU-per-worker assignment
applies as usual; `tla_ut/conftest.py` shadows it for no-NPU ir-only mode. The
per-case `python <case>.py` entry keeps working unchanged.
