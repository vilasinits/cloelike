"""Photometric angular power spectrum likelihoods with a selectable theory backend.

Same pipeline as ``EuclidLikelihood_photo_Cls`` -- rebinning, mixing matrices,
scale-cut masking, data vector, covariance and chi^2 all come from
``PhotoLikelihoodBase`` -- but the theory C_ell come from a backend object
(see ``cloelike.photo_theory_backends``):

* ``CloelibPhotoTheory``: cloelib, as in ``EuclidLikelihood_photo_Cls``
* ``GODMAXPhotoTheory``: the GODMAX halo model

Probes
------
======== ================ =========================================
name     key              bins
======== ================ =========================================
WL       SHE, SHE, i, j   i <= j
GCph     POS, POS, i, j   i <= j
GGL      POS, SHE, i, j   all
CMBL     CMBL, CMBL, 1, 1 single map
CMBLxGC  CMBL, POS, 1, j  all position bins
CMBLxWL  CMBL, SHE, 1, j  all shear bins
YY       Y, Y, 1, 1       single map
GCxY     POS, Y, i, 1     all position bins
WLxY     SHE, Y, i, 1     all shear bins
CMBLxY   CMBL, Y, 1, 1    single map pair
======== ================ =========================================

The data vector, mask and theory vector are concatenated in the order of
``probes``; the covariance must follow the same block order. ``data["cells"]``
and ``settings["scale_cuts"]`` need one entry per key. In ``mode="coupled"``
``data["mixmat"]`` needs one mixing matrix per key; its ``ell`` is the data
``ells``.

Example
-------
>>> theory = GODMAXPhotoTheory("params_godmax.yaml")
>>> like = EuclidLikelihood_Cls(data, settings, theory,
...                             probes=("WL", "GGL", "GCph", "CMBL", "CMBLxGC"))
>>> like.loglike(parameters)
"""

import numpy as np

from cloelike.EuclidLikelihood_photo_base import PhotoLikelihoodBase
from cloelike.photo_theory_backends import (
    CloelibPhotoTheory,
    GODMAXPhotoTheory,
    e_mode_slice,
    n_spin2_legs,
)

__all__ = [
    "PROBES",
    "apply_mixing_matrix",
    "EuclidLikelihood_Cls",
    "EuclidLikelihood_Cls_WL",
    "EuclidLikelihood_Cls_GCph",
    "EuclidLikelihood_Cls_GGL",
    "EuclidLikelihood_Cls_2x2pt",
    "EuclidLikelihood_Cls_3x2pt",
    "EuclidLikelihood_Cls_6x2pt",
    "CloelibPhotoTheory",
    "GODMAXPhotoTheory",
]

# name -> (field1, field2, upper triangle only when field1 == field2)
PROBES = {
    "WL": ("SHE", "SHE", True),
    "GCph": ("POS", "POS", True),
    "GGL": ("POS", "SHE", False),
    "CMBL": ("CMBL", "CMBL", True),
    "CMBLxGC": ("CMBL", "POS", False),
    "CMBLxWL": ("CMBL", "SHE", False),
    "YY": ("Y", "Y", True),
    "GCxY": ("POS", "Y", False),
    "WLxY": ("SHE", "Y", False),
    "CMBLxY": ("CMBL", "Y", False),
}


def apply_mixing_matrix(key, cl, mixing_matrix):
    """Convolve a full-sky C_ell (cloelib layout, ell = 0..n-1) with a mixing matrix.

    Same operations as ``AngularTwoPoint.get_pseudo_Cl``; spin-0 x spin-0 and
    spin-0 x spin-2 rules also cover the CMBL and Y fields. ``mixing_matrix`` is
    a euclidlib mixing matrix (``.array``) or, after rebinning, a plain array.
    """
    mm = np.asarray(getattr(mixing_matrix, "array", mixing_matrix))
    cl = np.asarray(getattr(cl, "array", cl))
    n_in = mm.shape[-1]
    legs = n_spin2_legs(key)
    if legs == 0:
        return mm @ cl[:n_in]
    if legs == 1:
        return np.stack([mm @ cl[0, :n_in], mm @ cl[1, :n_in]])
    ee, bb = cl[0, 0, :n_in], cl[1, 1, :n_in]
    out = np.empty((2, 2, mm.shape[-2]))
    out[0, 0] = mm[0] @ ee + mm[1] @ bb
    out[0, 1] = mm[2] @ cl[0, 1, :n_in]
    out[1, 0] = mm[2] @ cl[1, 0, :n_in]
    out[1, 1] = mm[0] @ bb + mm[1] @ ee
    return out


class EuclidLikelihood_Cls(PhotoLikelihoodBase):
    """Photometric C_ell likelihood for any combination of ``PROBES``.

    Parameters
    ----------
    data : dict
        ``cells``, ``ells``, ``z_arr``, ``cov``; ``dndz_she`` if any probe has a
        SHE field, ``dndz_pos`` if any has a POS field; ``mixmat`` in coupled mode.
    settings : dict
        ``scale_cuts`` (one ``[ell_min, ell_max]`` per key), optional ``n_ell_bins``.
    theory : CloelibPhotoTheory or GODMAXPhotoTheory
        Theory backend. Each likelihood configures its backend in ``__init__``;
        use one backend instance per likelihood.
    probes : sequence of str, optional
        Probe names, in data-vector order. Defaults to the class attribute.
    mode : str
        ``"coupled"`` (apply mixing matrices) or ``"decoupled"``.
    """

    probes = ()

    def __init__(self, data, settings, theory, probes=None, mode="coupled"):
        super().__init__(data, settings, None, None, None, mode=mode)
        self.probes = tuple(self.probes if probes is None else probes)
        if not self.probes:
            raise ValueError(f"No probes given; choose from {list(PROBES)}.")
        unknown = [p for p in self.probes if p not in PROBES]
        if unknown:
            raise ValueError(f"Unknown probes {unknown}; choose from {list(PROBES)}.")
        if len(set(self.probes)) != len(self.probes):
            raise ValueError(f"Duplicate probes in {self.probes}.")

        self.theory = theory
        self.blocks = [(name, self._probe_keys(name)) for name in self.probes]
        fields = {f for name in self.probes for f in PROBES[name][:2]}
        self.theory.setup(self.data, fields)

    def _n_bins(self, field):
        if field == "SHE":
            return self.data["dndz_she"].shape[0]
        if field == "POS":
            return self.data["dndz_pos"].shape[0]
        return 1

    def _probe_keys(self, name):
        field1, field2, upper = PROBES[name]
        n1, n2 = self._n_bins(field1), self._n_bins(field2)
        return [
            (field1, field2, i, j)
            for i in range(1, n1 + 1)
            for j in range(i if upper else 1, n2 + 1)
        ]

    @property
    def keys(self):
        """All probe keys, in data-vector order."""
        return [key for _, keys in self.blocks for key in keys]

    def get_masking_vector(self):
        ells = np.asarray(self.data["ells"])
        return np.concatenate(
            [self._masking(ells, self.scale_cuts[key]) for key in self.keys]
        )

    def get_data_vector_full(self):
        return np.concatenate(
            [np.ravel(e_mode_slice(key, self.data["cells"][key])) for key in self.keys]
        )

    def _theory_ells(self, keys):
        if self.mode != "coupled":
            return np.asarray(self.data["ells"])
        missing = [key for key in keys if key not in self.mixmat]
        if missing:
            raise KeyError(
                f"mode='coupled' needs data['mixmat'] entries for {missing}; "
                "provide them or use mode='decoupled'."
            )
        lmax = max(
            np.asarray(getattr(self.mixmat[k], "array", self.mixmat[k])).shape[-1]
            for k in keys
        )
        return np.arange(0, lmax)

    def get_theory_vector_full(self, parameters):
        vectors = []
        for _, keys in self.blocks:
            cells = self.theory.get_cells(parameters, keys, self._theory_ells(keys))
            if self.mode == "coupled":
                cells = {
                    key: apply_mixing_matrix(key, cells[key], self.mixmat[key])
                    for key in keys
                }
            self.theory_prediction.update(cells)
            vectors.extend(np.ravel(e_mode_slice(key, cells[key])) for key in keys)
        self.derived.update(self.theory.derived)
        return np.concatenate(vectors)


class EuclidLikelihood_Cls_WL(EuclidLikelihood_Cls):
    """Cosmic shear."""

    probes = ("WL",)


class EuclidLikelihood_Cls_GCph(EuclidLikelihood_Cls):
    """Photometric galaxy clustering."""

    probes = ("GCph",)


class EuclidLikelihood_Cls_GGL(EuclidLikelihood_Cls):
    """Galaxy-galaxy lensing."""

    probes = ("GGL",)


class EuclidLikelihood_Cls_2x2pt(EuclidLikelihood_Cls):
    """GGL + GCph, in the data-vector order of ``EuclidLikelihood_2x2pt``."""

    probes = ("GGL", "GCph")


class EuclidLikelihood_Cls_3x2pt(EuclidLikelihood_Cls):
    """WL + GGL + GCph, in the data-vector order of ``EuclidLikelihood_3x2pt``."""

    probes = ("WL", "GGL", "GCph")


class EuclidLikelihood_Cls_6x2pt(EuclidLikelihood_Cls):
    """3x2pt + CMB lensing auto and its crosses with positions and shear."""

    probes = ("WL", "GGL", "GCph", "CMBL", "CMBLxGC", "CMBLxWL")
