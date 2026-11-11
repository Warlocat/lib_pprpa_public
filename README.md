lib_pprpa
======
Library for particle-particle random phase approximation.

Installation
--------

### Prerequisites

- [PySCF](https://github.com/pyscf/pyscf) 2.7 or higher, and all dependencies

#### I don't want to develop the code. I just want to install it.

Clone this repo and run `pip install .` in your Python environment.

#### Method 1. Set PYTHONPATH
After cloning the repo, you can set the environment variable `PYTHONPATH` so that the Python interpreter can find `lib_pprpa`.

For example, if you cloned the repository in the folder `/opt`, you could run
```
export PYTHONPATH=/opt/lib_pprpa:$PYTHONPATH
```

or put this line in your `.bashrc` so that it runs at the start of every session.

#### Method 2. Pip editable installation
If you want to install different versions of `lib_pprpa` in separate Python environments, or if you don't like environment variables, you can do an editable installation with Pip. This method also works for most Python packages in the wild. Clone the repo and run
```
pip install -e .
```
No dependencies will be installed in this step---you have to install them separately.

Method 2 is recommended for new users, as it is [standard practice](https://docs.pytest.org/en/7.1.x/explanation/goodpractices.html).
Consult [the pip documentation](https://pip.pypa.io/en/stable/cli/pip_install/) for further information
about command-line options for pip.

#### GPU support
The GPU modules (`lib_pprpa.gpu_*`, `lib_pprpa.pprpa_eri_gpu`, `lib_pprpa.grad.pprpa_gamma_gpu`,
`lib_pprpa.grad.pprpa_gamma_gdf_gpu`) need an NVIDIA GPU and [GPU4PySCF](https://github.com/pyscf/gpu4pyscf)
at commit `d65bd284` (with the GDF-gradient memory fix `a66c7c4` cherry-picked), built from source:
no release has the APIs they use, and a `pip install` from git leaves out the periodic GDF
derivatives. `examples/gpu/README.md` has the build. The
extras install everything else: `gpu-cuda13` (CuPy, GPU4PySCF's libxc and the CUDA 13 libraries
as wheels, which cuBLAS's fp64 emulation in `lib_pprpa.gpu_emulation` needs) or `gpu` (CUDA 12
toolkit on the system; those GEMMs then run in plain fp64):
```
pip install -e ".[gpu-cuda13]"   # CUDA 13
pip install -e ".[gpu]"          # CUDA 12
```
Check the installation with
`python -c "import gpu4pyscf, cupy; print(cupy.cuda.runtime.getDeviceCount())"` and run the GPU
tests with `pytest tests`.

Features
--------

* ppRPA excitation energy
  Spin-restricted (spin-adapted), spin-unrestricted, generalized, fractional-charge
  Analysis tools: density matrix, natural transition orbital

* ppRPA analytic gradient
  Spin-restricted (spin-adapted), generalized
  Three-dimensional real Gamma-point periodic GDF energy/operator/gradient
  (GPU4PySCF derivative back end; see `examples/gpu/README.md`)

* T-matrix for quasiparticle energy
  Spin-restricted

References
----------

Please cite the following papers in publications utilizing the lib_pprpa package:

* J. Chem. Phys. 164, 144118 (2026)

* Phys. Rev. A 88, 030501(R)

* J. Phys. Chem. A 2023, 127, 37, 7811–7822

Cite the following paper if T-matrix code is used:

* J. Phys. Chem. Lett. 2021, 12, 26, 6203–6210

* J. Phys. Chem. Lett. 2017, 8, 14, 3223–3227
