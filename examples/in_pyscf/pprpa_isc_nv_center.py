'''
This example demonstrates how to calculate ISC rates using ppRPA gradients.
Please check pprpa_isc.py for a simpler example and more details on the workflow.
'''
from pyscf import scf, gto, dft
import numpy, h5py
from socutils.somf import somf_pt

mol = gto.Mole()
mol.atom = '''
C    0.00000000  0.00000000  0.00000000
C    3.56828513 -3.56666261  0.00162245
C    1.79848966 -0.00993297 -1.76589334
C    1.78613682  0.01061686  1.78133366
C    5.35189073  0.04522721 -1.78291211
C    5.33337802 -0.00827688  1.77050681
C    0.90085001  0.88741486 -0.88744514
C    4.45569387 -2.66581261 -0.88582880
C    4.45412584  0.88907956  2.66735036
C    1.78692689 -1.78055017 -0.00897024
C    1.79770601  1.76669143  0.00987495
C    5.33415428 -1.76814880  0.01148293
C    5.35112322  1.78525175 -0.04372605
C    0.89492982 -0.88965798  0.89048182
C    4.45014966 -0.87317297 -2.66032334
C    4.51587946 -0.96655028  0.96900913
C    4.44859001  2.66270742  0.87385320
C    3.56824133 -0.00004427 -3.56780297
C    3.56666689  3.56937691 -0.00010180
C    7.13603957  0.00162724  0.00152114
C    2.67781533 -2.67173281  0.89129248
C    2.69429899  0.88187930 -2.66112170
C    2.59924087  0.94921681  0.96813766
C    6.22857170  0.88353511  0.87466253
C    3.55833767 -1.76898077 -1.76509317
C    3.57727950 -1.77971139  1.78214807
C    3.61188983  1.78443692 -1.78370326
C    3.55677775  1.76751556  1.76969901
C    2.67702891 -0.88882307  2.67334308
C    2.60008629 -0.96744782 -0.94760338
C    2.69351385  2.66188517 -0.88197358
C    6.22934620 -0.87233942 -0.88036588
N    4.55433966  0.98767703 -0.98614334
C    3.56671117  0.00167100  3.56823657
H   -0.61776034 -0.61601297 -0.61798334
H   -0.61830465  0.61800744  0.61544243
H    4.18629363 -4.18440617  0.61762701
H    2.95283430 -4.18498404 -0.61635984
H    1.17701966 -0.61225297 -2.39508334
H    1.16152185  0.61501673  2.40539857
H    5.97702322  0.64796721 -2.40805962
H    5.96258415 -0.61001094  2.39252783
H    0.28833013  1.50750016 -1.50809970
H    5.07633989 -3.27833247 -1.50592266
H    5.07422792  1.50974235  3.27984492
H    1.16285403 -2.40518184 -0.61334464
H    1.17569279  2.39589730  0.61161738
H    5.96336912 -2.38960205  0.61379426
H    5.97624583  2.41038454 -0.64649155
H    0.28443981 -1.51101290  1.51186689
H    5.07242966 -1.50442297 -3.25965334
H    6.82923784 -1.50359991 -1.50209382
H    5.07032696  3.26261610  1.50508862
H    4.18300336  0.61471369 -4.19153297
H    2.95666366 -0.61218647 -4.19721285
H    2.95454157  4.19881175  0.61146707
H    4.18142498  4.19308183 -0.61488916
H    7.76547441 -0.60993355  0.61365454
H    7.75974449  0.61638940 -0.61326215
H    2.05645547 -3.28222281  1.51267260
H    2.06303880  1.50359950 -3.26102167
H    2.95071500  0.61967953  4.18598848
H    6.82791040  1.50583184  1.50588777
H    2.95658659 -2.39101073 -2.39427420
H    4.18167938 -2.40376753  2.40677179
H    4.21462983  2.40955923 -2.40886096
H    2.95447476  2.39673065  2.39116009
H    2.05567384 -1.51018308  3.28385844
H    4.18471865 -0.61376302  4.18654956
H    2.06226354  3.26120683 -1.50426129
'''
mol.basis = 'cc-pvdz'
mol.charge = -3
mol.verbose = 4
mol.max_memory = 170000
mol.build()
nao = mol.nao

# ppTDA and ppRPA give similar SOC matrix elements
nroots = 3
nocc = mol.nelectron//2
nvir = 0 #mol.nao_nr() - nocc


chkfname = 'scf.chk'
mf = dft.RKS(mol, xc="b3lyp").sfx2c1e().density_fit()
mf.chkfile=chkfname
# mf.__dict__.update(scf.chkfile.load(chkfname, 'scf'))
mf.kernel()

mo_coeff_rhf = mf.mo_coeff[:,mol.nelectron//2-nocc : mol.nelectron//2+nvir]
soints_ao = somf_pt.get_soc_mf_bp(mf)
with h5py.File("soc.h5", "w") as f:
  f.create_dataset("soc2c", shape=soints_ao.shape, data=soints_ao)
with h5py.File("soc.h5", "r") as f:
    soints_ao = f["soc2c"][()]
soints_mo = numpy.array([mo_coeff_rhf.conj().T @ soints_ao[i] @ mo_coeff_rhf for i in range(3)])

from lib_pprpa import pyscf_util,pprpa_davidson
nocc_act, mo_energy_act, Lpq = pyscf_util.get_pyscf_input_mol(mf, nocc_act=nocc, nvir_act=nvir)
pprpa = pprpa_davidson.ppRPA_Davidson(nocc_act, mo_energy_act, Lpq, channel="hh", nroot=nroots)
pprpa.kernel("s")
pprpa.kernel("t")
pprpa.analyze()


from lib_pprpa.grad.grad_utils import get_isc_S_T
soc_matrix = get_isc_S_T(mf, pprpa, socints_mo=soints_mo, calculate_tt=True)

