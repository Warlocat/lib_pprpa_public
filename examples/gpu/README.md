# GPU Gamma-point pp-RPA — requirements

This folder has three GPU periodic examples:

- **`pbc_pprpa_gamma_opt_gpu.py`** — small periodic cell, **AO-direct** energy path.
  Library-only (lib_pprpa + gpu4pyscf + ase). Runs in a few minutes on one GPU.
- **`pbc_pprpa_gamma_opt_nv_gpu.py`** — large defect cell (NV-center style), **MO-eri**
  energy path with a frozen-core active space. Tunable `ke_cutoff` on the command
  line. This is the path used for the 63-atom NV production runs.
- **`pbc_pprpa_gamma_gdf_gpu.py`** — small one-step **GDF-consistent** ppRPA
  energy and analytical gradient. The reference, Davidson operator, CPHF
  response, and every two-electron derivative all use periodic GDF.

The two optimization examples share the same FFT/AFT back end:

```
GPU KRKS SCF (gpu4pyscf)
  -> [energy solve: AO-direct OR MO-eri — see below]
  -> GPU Gamma-point pp-RPA gradient    (lib_pprpa.grad.pprpa_gamma_gpu)
  -> ASE BFGS                           (lib_pprpa.grad.ase_utils.kernel)
```

The energy solve differs:

```
AO-direct (small):  lib_pprpa.pprpa_davidson_gpu.attach_gpu_contraction(mp, kg)
MO-eri    (large):  lib_pprpa.gpu_ao2mo.gpu_ao2mo_blocks  -> active-space vvvv/oovv/oooo
                    lib_pprpa.pprpa_eri_gpu.attach_gpu_eri_contraction(mp, ...)
```

## Software required
- **gpu4pyscf** at upstream commit `d65bd284b4802fc5081b199dec710a5f2d0c56ef`
  (between the 1.8.1 release and current master; both FFTDF and GDF paths are
  validated there) with the upstream GDF-gradient memory fix `a66c7c4`
  cherry-picked, built from source as below.  No release has the APIs both
  need: the 1.8.1 wheels lack the local-pseudopotential gradient kernel and the
  periodic GDF derivatives (`gpu4pyscf.pbc.df.grad`), and later master changed
  the J/K derivative and three-center integral interfaces.
- **pyscf** (the periodic `gto`/`dft` driver and `lib_pprpa`).
- **ase** (the BFGS optimizer used by `ase_utils.kernel`).
- **lib_pprpa** on `PYTHONPATH`.
- A GPU (the small example fits in any modern GPU; NV-scale needs ~32-80 GB).

## Restarts

A rerun of either optimization script in the same directory resumes: the
geometry comes from the last frame of `opt.traj`, the BFGS Hessian from
`bfgs.pkl` (ASE's own restart file; `ase_utils.kernel` passes `restart=`,
`trajectory=` and `append_trajectory=` through), and every SCF starts from the
orbitals of the previous geometry in `scf.chk` (PySCF's checkpoint).  The CPHF
solution of the previous step seeds the next one within a run
(`Gradients.cphf_x0`).  ASE writes the Hessian before the trajectory frame, so
a kill between the two repeats one step on resume.  Delete the three files to
start over.

## gpu4pyscf source patch (hybrid functionals at large systems)
The exchange terms of the pp-RPA gradient (the 2-RDM exchange of the relaxed
density and the pairing force) are built from the low-rank factors of the pair
densities and need no patch.  Only the hybrid reference exchange force still
goes through the AFT kernel below, whose fixed block size OOMs at large `nao` /
fine mesh.  It is a **pure-Python edit — no rebuild needed**, but a
`git checkout`/reinstall of gpu4pyscf will wipe it.

### `gpu4pyscf/pbc/df/aft_jk.py` — `get_ek_ip1` block size (~line 522)
The K energy-gradient kernel under-budgets its `nao^2 * blk` arrays. Change the
divisor factor `*2` -> `*8`:
```python
# from:
blksize = int(avail_mem/(nao**2*bvk_ncells*16*2))//16*16
# to:
blksize = int(avail_mem/(nao**2*bvk_ncells*16*8))//16*16
```

Keep an `.orig_bak` backup next to the file; to revert, copy it back.

## MO-eri version (`pbc_pprpa_gamma_opt_nv_gpu.py`) — what it additionally needs

The AO-direct contraction rebuilds the pairing kernel from the AOs every Davidson
matvec; at large active spaces (AS=100) that is the bottleneck. The MO-eri path
instead forms the active-space MO integrals once per geometry and contracts them
on the GPU — about 4 orders of magnitude faster per Davidson solve (≈0.3 s vs
≈90 min at AS=100). What it requires on top of the AO-direct example:

1. **Two extra lib_pprpa modules** (now shipped in the package):
   - `lib_pprpa.gpu_ao2mo` — `gpu_ao2mo_blocks(cell, cocc, cvir, mesh, …)` builds the
     active-space `vvvv` / `oovv` / `oooo` blocks via GPU FFT ao2mo as packed pair
     Gram matrices (the lower triangle of `(pq|rs)` over `p >= q` pairs; 8x smaller
     than the physicist `vvvv`, 2x for `oovv`), assembled in pinned host memory.
   - `lib_pprpa.pprpa_eri_gpu` — `attach_gpu_eri_contraction(mp, vvvv, oovv, oooo)`
     swaps the Davidson matvec to GEMMs with the pp-RPA pair-space operator, built
     from the Gram blocks at the first product (or straight from ao2mo with
     `gpu_ao2mo_pair_operator(..., multi)`); `pack_eri` converts physicist tensors.
2. **Memory at scale.** The packed blocks stay on the device when they fit 75%
   of its memory and are otherwise streamed in row chunks per Davidson iteration;
   the 216-atom NV cell at AS=300 (49 GB packed) is resident on one H200 or B200.
3. **A geometry file** (POSCAR/`.vasp` or `.xyz`) — read via ASE; the lattice is
   taken from it and held fixed during the relaxation.
4. **Enough GPU memory.** The peak is the FFT ao2mo, not the stored ERIs: the
   `vvvv` codensity array is `nvir²·ngrid·8 B`, e.g. AS=100 on the NV cell
   (nvir=100, mesh 59³ ⇒ ngrid≈2.05e5) is ≈16 GB for that one intermediate, plus
   the FFT transients. A 32 GB V100 OOMs at AS=100; use an **A100-80g**
   (`--constraint=a100-80g` on Grace). The example frees the SCF GPU pool before
   ao2mo. To shrink the peak on a smaller GPU, lower `AS` or pass a smaller
   `pair_blk` to `gpu_ao2mo_blocks`.

Tunable knobs exposed on the command line: `ke_cutoff` (plane-wave cutoff),
`mult` (s/t), `AS` (active-space size), `fmax`, `maxsteps`. The defect defaults
(NV⁻-style: `charge=-3`, hh channel, gth-dzvp/gth-pbe, PBE) are set near the top
of the script — edit there for a different defect or charge state.

## Environment notes
- Load CUDA (we used `CUDA/12.8.0`) and the gpu4pyscf conda env before running.
  Bouchet: `module purge && module load CUDA/12.8.0 && source ~/.bash_gpu4pyscf`.
  Grace: `module load miniconda CUDA/12.8.0 imkl/2024.2.0 && source activate gpu4pyscf`,
  with `PYTHONPATH=~/project/gpu4pyscf:~/project/pyscf:~/project/lib_pprpa_public`
  and `CUPY_ACCELERATORS=cutensor,cub`.
- Run on a GPU node (login nodes have no GPU).

### CUDA 13 venv with gpu4pyscf d65bd28
The MO-eri path runs its widest DGEMMs (the ao2mo tiles, and the CPHF response
on Blackwell) as cuBLAS fp64 emulation where it is faster (`lib_pprpa.gpu_emulation`;
on by default through `emulate=True` in `gpu_ao2mo_blocks` and `make_gpu_vresp`).
That needs cuBLAS 13; the pip CUDA 13 wheels ship it with the rest of the CUDA
libraries and the compiler, so no CUDA module is needed to build or run.
gpu4pyscf is not installed from PyPI but built from the pinned commit; a
`pip install git+...` of it does not work either: its `setup.py` drops
`gpu4pyscf.pbc.df.grad` (no `__init__.py` at that commit), the default
architectures include sm_70, which CUDA 13 no longer compiles, and the CMake
project needs a Fortran compiler.  Build it in place and put the checkout on
the venv's path:
```bash
uv venv --python 3.10 ~/venv_cu13
uv pip install --python ~/venv_cu13/bin/python     cupy-cuda13x==14.2.0 gpu4pyscf-libxc-cuda13x==0.8.1 pyscf==2.14.0     pyscf-dispersion geometric packaging 'numpy<=2.3' h5py scipy ase cutensor-cu13     nvidia-cublas nvidia-cuda-runtime nvidia-cuda-nvrtc nvidia-nvjitlink     nvidia-cufft nvidia-curand nvidia-cusolver nvidia-cusparse
# the compiler wheels at the installed runtime's version, CUB headers and CMake
RT=$(uv pip list --python ~/venv_cu13/bin/python | awk '$1=="nvidia-cuda-runtime"{print $2}')
uv pip install --python ~/venv_cu13/bin/python nvidia-cuda-nvcc==$RT nvidia-cuda-crt==$RT     nvidia-nvvm==$RT nvidia-cuda-cccl 'cmake>=3.21,<4'

git clone https://github.com/pyscf/gpu4pyscf ~/gpu4pyscf_d65
git -C ~/gpu4pyscf_d65 checkout -b d65bd28-a66c7c4 d65bd284b4802fc5081b199dec710a5f2d0c56ef
git -C ~/gpu4pyscf_d65 cherry-pick -x a66c7c4bd84509c585c3326f15346de755af7b41
SP=~/venv_cu13/lib/python3.10/site-packages
CU=$SP/nvidia/cu13
PATH=$CU/bin:~/venv_cu13/bin:$PATH cmake -S ~/gpu4pyscf_d65/gpu4pyscf/lib -B ~/gpu4pyscf_d65/build     -DBUILD_LIBXC=OFF -DCUDA_ARCHITECTURES="80-real;90-real;100-real" -DCMAKE_BUILD_TYPE=Release     -DCMAKE_CUDA_COMPILER=$CU/bin/nvcc -DCUDAToolkit_ROOT=$CU     -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++     -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ -DCMAKE_Fortran_COMPILER=$(command -v gfortran)
PATH=$CU/bin:~/venv_cu13/bin:$PATH cmake --build ~/gpu4pyscf_d65/build -j 12
echo ~/gpu4pyscf_d65 > $SP/gpu4pyscf_d65.pth
ln -s libcusolver.so.12 $CU/lib/libcusolver.so

uv pip install --python ~/venv_cu13/bin/python --no-deps -e /path/to/lib_pprpa
export LD_LIBRARY_PATH=$CU/lib:$SP/cutensor/lib:$LD_LIBRARY_PATH
```
- The cherry-pick is upstream #907 (Python only), the fix for the GDF
  gradient running out of device memory in the pair-density exchange: without
  it the 63-atom NV cell (gth-dzvp, naux 5292) OOMs even on a 180 GB B200.
- `CUDA_ARCHITECTURES` lists the GPUs to run on: `80` A100, `90` H100/H200,
  `100` B200.  The build takes about 10 min on 12 cores.
- CMake's `project()` declares Fortran although only the bundled libxc (off
  here; the `gpu4pyscf-libxc-cuda13x` wheel provides it) uses it, so a
  `gfortran` must be on the path (on a cluster, from a GCC module).  Keep the
  system `gcc`/`g++` so the libraries need only the system `libstdc++`.
- `nvidia-cuda-cccl` provides `cub/cub.cuh`, which the `nvcc` wheel lacks.
- gpu4pyscf loads cuSOLVER through `find_library('cusolver')`, which needs the
  unversioned `libcusolver.so` on `LD_LIBRARY_PATH`; the wheel ships only
  `libcusolver.so.12`, hence the link.
- Install `lib_pprpa` with `--no-deps` (or with the `gpu-cuda13` extra, which
  lists the wheels above but not gpu4pyscf): the checkout is not a pip
  distribution, so a `gpu4pyscf` requirement would pull a release next to it.
- List the `nvidia-*` runtime wheels explicitly: installing only cupy leaves out
  cuSOLVER and nvJitLink, `import gpu4pyscf` fails, and pytest then skips
  every GPU test.
- Do not set `CUBLAS_EMULATE_DOUBLE_PRECISION`: it emulates every DGEMM in the
  process and slows the SCF and CPHF.  Leave `CUPY_ACCELERATORS` unset as well
  (cuTENSOR 2.8.1 rejects a strided accumulate in `gpu_pairing_force`).
- Under CUDA 12 the switch is a no-op, so the same code runs with a CUDA 12
  build of the same commit.  Hopper gains only on very wide tiles; Blackwell
  gains from ~800-pair strips up.

## Scope / notes
- Gamma point, RKS/RHF reference, LDA/GGA/hybrid functionals.
- The RKS CPHF/Z-vector solve uses native
  `KRKS.gen_response(singlet=None, hermi=1)`, which caches the XC kernel once
  for all response applications.  The ppRPA-specific nuclear XC-gradient
  skeleton remains in `lib_pprpa.grad.grad_utils_gpu_pbc`.
- The example uses the **AO-direct** GPU Davidson (library-only). At large active
  spaces the **MO-eri** path (GPU FFT ao2mo + batched `use_eri` contraction) is
  much faster — that is the path used for the NV-center production runs.
- The CPU `make_rdm1_relaxed_rhf_pprpa` builds the relaxed density (small MO-space
  algebra); the 2-RDM exchange (`gpu_fft_k.pair_get_k_lowrank`, from the low-rank
  factors of the pair densities) and the response run on the GPU.

## Fully GDF-consistent path

`pbc_pprpa_gamma_gdf_gpu.py` is a separate backend. Construct a converged CPU
PySCF periodic GDF reference, call
`lib_pprpa.pbc_gdf.make_pprpa_gdf(mf, ...)`, and evaluate the state with
`lib_pprpa.grad.pprpa_gamma_gdf_gpu.Gradients`. The solver records the exact
SCF orbitals, auxiliary basis, metric threshold, mesh, and GDF object identity;
the gradient fails rather than silently mixing a different reference or an
FFT/AFT exchange derivative.

Current scope is a real restricted three-dimensional Gamma-point reference,
the pp/hh Davidson Lpq operator, HF or LDA/GGA/global-hybrid KS, and a full
J+K GDF build. Range-separated hybrids and meta-GGAs are rejected. This path
uses GPU4PySCF's internal periodic GDF derivative interface, including
`hermi=2` for the antisymmetric pair density. Use the exact compatible
GPU4PySCF revision cited in the validation report; the driver checks the
required call signature at runtime. GDF auxiliary-basis convergence must be
checked for the target system because it changes both the reference and ppRPA
Hamiltonian.
