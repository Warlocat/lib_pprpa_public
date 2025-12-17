import numpy as np
from pyscf.pbc import gto
from lib_pprpa.grad import ase_utils

cell = gto.M(
    atom='''
O 0.000000000000   0.000000000000   0.000000000000
O 1.1 0.0 0.0
''',
    basis='gth-szv',
    pseudo='gth-pbe', # pseudopotential is required for PBC pprpa optimization
    a=np.eye(3)*5.0,
    charge=2,
    verbose=4,
)

ase_utils.kernel(cell, xc="pbe", channel="pp", mult='t', state=0)