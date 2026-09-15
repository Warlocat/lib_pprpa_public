# Periodic GDF validation

These scripts validate the separate, fully Gaussian-density-fitted (GDF)
Gamma-point ppRPA path. Run them on a GPU node with an ABI-matched current
GPU4PySCF build and this repository first on `PYTHONPATH`.

The implementation was last validated against GPU4PySCF commit
`d65bd284b4802fc5081b199dec710a5f2d0c56ef`. It intentionally calls the
internal periodic GDF derivative kernel because the public wrapper does not
expose the required antisymmetric (`hermi=2`) contraction; a runtime signature
guard rejects incompatible revisions.

`validate_end_to_end.py` compares the complete analytical state gradient with
central finite differences of the same CPU-GDF reference and Lpq Davidson
operator. It covers both symmetric singlet and antisymmetric triplet pair
densities. The compact auxiliary basis is intentionally only an algebraic
regression case; it is not a physical production recommendation.

```bash
python validate_end_to_end.py --system c2 --auxbasis compact --ke-cutoff 100 \
  --mult s t --steps 0.002 0.004 --output gdf_fd.json
```

The default is the particle-particle channel; repeat the gate with
`--channel hh` to exercise the hole-hole density/sign path.

`validate_auxbasis_sensitivity.py` measures how the reference, ppRPA
excitation, and total state energies change along an even-tempered auxiliary-
basis sequence. Production work must repeat this convergence check for its own
cell, orbital basis, active space, and target states.

```bash
python validate_auxbasis_sensitivity.py --include-weigend \
  --output gdf_auxbasis.json
```

The implementation rejects non-Gamma, complex/unrestricted, non-3D, J-only,
range-separated-hybrid, and meta-GGA cases rather than falling back to a
different integral backend.
