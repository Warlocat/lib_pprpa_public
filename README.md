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

Features
--------

* ppRPA excitation energy
  Spin-restricted (spin-adapted), spin-unrestricted, generalized, fractional-charge
  Analysis tools: density matrix, natural transition orbital

* ppRPA analytic gradient
  Spin-restricted (spin-adapted), generalized

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
