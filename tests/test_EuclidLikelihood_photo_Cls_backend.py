"""Tests for EuclidLikelihood_photo_Cls_backend and photo_theory_backends.

* The cloelib backend reproduces EuclidLikelihood_3x2pt (decoupled and coupled)
  on synthetic inputs, without network access.
* The GODMAX backend: parameter translation, the CMB-lensing Limber path checked
  against GODMAX's own C_ell matrices, and likelihoods over every probe (skipped
  when godmax is not installed).
"""

import copy

import numpy as np
import pytest

from cloelib.cosmology.camb_cosmology import (
    CAMBBackground,
    CAMBLinearPerturbations,
    CAMBNonLinearPerturbations,
)
from cloelike.EuclidLikelihood_photo_Cls import EuclidLikelihood_3x2pt
from cloelike.EuclidLikelihood_photo_Cls_backend import (
    PROBES,
    CloelibPhotoTheory,
    EuclidLikelihood_Cls,
    EuclidLikelihood_Cls_3x2pt,
    GODMAXPhotoTheory,
)
from cloelike.photo_theory_backends import e_mode_slice

N_BINS = {"SHE": 2, "POS": 2, "CMBL": 1, "Y": 1}
Z = np.linspace(0.05, 1.5, 31)
DNDZ = np.array([np.exp(-0.5 * ((Z - mean) / 0.15) ** 2) for mean in (0.5, 0.9)])
DNDZ /= np.trapezoid(DNDZ, Z, axis=1)[:, None]
ELLS = np.array([20.0, 40.0, 80.0, 160.0, 320.0, 640.0])
ELL_MAX_MIXING = 700


class FakeMixingMatrix:
    """Minimal stand-in for a euclidlib mixing matrix."""

    def __init__(self, array, ell):
        self.array = array
        self.ell = ell
        self.lower = None
        self.upper = None

    @property
    def shape(self):
        return self.array.shape

    def __matmul__(self, other):
        return self.array @ other


def probe_keys(probes):
    keys = []
    for name in probes:
        field1, field2, upper = PROBES[name]
        for i in range(1, N_BINS[field1] + 1):
            for j in range(i if upper else 1, N_BINS[field2] + 1):
                keys.append((field1, field2, i, j))
    return keys


def n_shear_legs(key):
    return sum(field == "SHE" for field in key[:2])


def make_data(probes, mode, seed=1):
    rng = np.random.default_rng(seed)
    keys = probe_keys(probes)
    shapes = {2: (2, 2, len(ELLS)), 1: (2, len(ELLS)), 0: (len(ELLS),)}
    data = {
        "cells": {key: np.zeros(shapes[n_shear_legs(key)]) for key in keys},
        "ells": ELLS.copy(),
        "z_arr": Z,
        "dndz_she": DNDZ,
        "dndz_pos": DNDZ,
        "cov": np.eye(len(keys) * len(ELLS)) * 1e-18,
    }
    if mode == "coupled":
        mixmat = {}
        for key in keys:
            shape = (len(ELLS), ELL_MAX_MIXING)
            if n_shear_legs(key) == 2:
                shape = (3,) + shape
            array = rng.uniform(0.0, 1.0, shape) / ELL_MAX_MIXING
            mixmat[key] = FakeMixingMatrix(array, ELLS.copy())
        data["mixmat"] = mixmat
    return data


def make_settings(probes, include_rsd=False):
    return {
        "scale_cuts": {key: [0.0, 1e5] for key in probe_keys(probes)},
        "include_rsd": include_rsd,
    }


@pytest.fixture(scope="module")
def cloelib_params():
    return {
        "H0": 67.0,
        "Omega_cdm0": 0.26,
        "Omega_b0": 0.049,
        "Omega_k0": 0.0,
        "w0": -1.0,
        "wa": 0.0,
        "ns": 0.965,
        "As": 2.1e-9,
        "mnu": 0.06,
        "gamma_MG": 0.545,
        "N_mnu": 1,
        "alpha_s": 0.0,
        "log10TAGN": 7.75,
        "AIA": 0.5,
        "EtaIA": 0.2,
        "CIA": 0.0134,
        "b1_photo_poly0": 1.0,
        "b1_photo_poly1": 0.5,
        "b1_photo_poly2": 0.0,
        "b1_photo_poly3": 0.0,
        **{f"magnification_bias_{i}": 0.0 for i in (1, 2)},
        **{f"dz_pos_{i}": 0.01 for i in (1, 2)},
        **{f"dz_shear_{i}": -0.01 for i in (1, 2)},
        **{f"width_pos_{i}": 1.0 for i in (1, 2)},
        **{f"width_shear_{i}": 1.0 for i in (1, 2)},
        **{f"multiplicative_bias_{i}": 0.01 for i in (1, 2)},
    }


def cloelib_theory(include_rsd=False):
    return CloelibPhotoTheory(
        CAMBBackground,
        CAMBLinearPerturbations,
        CAMBNonLinearPerturbations,
        include_rsd=include_rsd,
    )


@pytest.mark.parametrize(
    ("mode", "include_rsd"),
    [("decoupled", False), ("coupled", False), ("decoupled", True)],
)
def test_cloelib_backend_reproduces_existing_3x2pt(mode, include_rsd, cloelib_params):
    probes = ("WL", "GGL", "GCph")
    data = make_data(probes, mode)
    settings = make_settings(probes, include_rsd)

    existing = EuclidLikelihood_3x2pt(
        copy.deepcopy(data),
        settings,
        CAMBBackground,
        CAMBLinearPerturbations,
        CAMBNonLinearPerturbations,
        mode=mode,
    )
    backend = EuclidLikelihood_Cls_3x2pt(
        copy.deepcopy(data), settings, cloelib_theory(include_rsd), mode=mode
    )

    np.testing.assert_array_equal(
        backend.get_masking_vector(), existing.get_masking_vector()
    )
    theory_existing = np.asarray(existing.get_theory_vector_full(cloelib_params))
    theory_backend = backend.get_theory_vector_full(cloelib_params)
    assert np.all(np.isfinite(theory_backend))
    np.testing.assert_allclose(theory_backend, theory_existing, rtol=1e-8, atol=1e-30)
    assert backend.loglike(cloelib_params) == pytest.approx(
        float(existing.loglike(cloelib_params)), rel=1e-8
    )
    assert float(backend.derived["sigma8_0"]) == pytest.approx(
        float(existing.derived["sigma8_0"])
    )


def test_cloelib_backend_cmb_lensing_probes(cloelib_params):
    probes = ("CMBL", "CMBLxGC", "CMBLxWL")
    data = make_data(probes, "decoupled")
    like = EuclidLikelihood_Cls(
        data, make_settings(probes), cloelib_theory(), probes=probes, mode="decoupled"
    )
    vector = like.get_theory_vector_full(cloelib_params)
    assert vector.shape == (len(probe_keys(probes)) * len(ELLS),)
    assert np.all(np.isfinite(vector))
    assert np.all(like.theory_prediction[("CMBL", "CMBL", 1, 1)] > 0)


def test_cloelib_backend_rejects_tsz():
    probes = ("GCxY",)
    with pytest.raises(ValueError, match="cannot compute fields"):
        EuclidLikelihood_Cls(
            make_data(probes, "decoupled"),
            make_settings(probes),
            cloelib_theory(),
            probes=probes,
            mode="decoupled",
        )


def test_unknown_probe_rejected():
    with pytest.raises(ValueError, match="Unknown probes"):
        EuclidLikelihood_Cls(
            make_data(("WL",), "decoupled"),
            make_settings(("WL",)),
            cloelib_theory(),
            probes=("WL", "kappa"),
            mode="decoupled",
        )


# --------------------------------------------------------------------------- GODMAX

GODMAX_CONFIG = {
    "sim_params": {
        "cosmo": {
            "flat": True,
            "H0": 67.2,
            "Om0": 0.31,
            "Ob0": 0.049,
            "sigma8": 0.81,
            "ns": 0.95,
            "w0": -1.0,
        },
        "theta_ej_0": 1.0,
        "log10_Mstar0_theta_ej": 16.0,
        "theta_co_0": 0.05,
        "log10_Mstar0_theta_co": 16.0,
        "mu_beta": 0.6,
        "log10_Mstar0": 14.0,
        "log10_Mc0": 14.83,
        "gamma_rhogas": 2.0,
        "delta_rhogas": 7.0,
        "a_zeta": 0.3,
        "n_zeta": 2.0,
        "alpha_nt": 0.18,
        "beta_nt": 0.5,
        "n_nt": 0.3,
    },
    "halo_params": {
        "rmin": 0.005,
        "rmax": 8.0,
        "nr": 16,
        "zmin": 0.01,
        "zmax": 1.6,
        "nz": 6,
        "lg10_Mmin": 10.5,
        "lg10_Mmax": 16.5,
        "nM": 8,
        "kmin": 1e-4,
        "kmax": 150.0,
        "nk": 24,
        "ellmin": 2.0,
        "ellmax": 3000.0,
        "nell": 16,
        "mdef_Delta": 200,
        "conc_model": "Duffy08",
        "hmf_model": "T10",
    },
    "analysis": {
        "backreaction": True,
        "baryonification_tSZ": True,
        "beam_fwhm_arcmin": 1.6,
        "num_points_trapz_int": 16,
        "nbar_gal_comoving_zarray": [0.01, 2.0, 64],
        "nbar_gal_comoving_val": 5e-4,
        "zmin_for_Cls": 0.01,
        "zmax_for_Cls": 1.6,
        "nz_for_Cls": 48,
        "split_cga_fftlog": True,
        "lowpass_Pmm1h_lowk": True,
        "kthresh_lowpass_Pmm1h_lowk": 1e-3,
    },
    "other_params": {
        "A_IA": 0.1,
        "eta_IA": 0.1,
        "z0_IA": 0.62,
        "C1_rhocrit": 0.0134,
        "Delta_z_bias_array": [0.0],
        "mult_shear_bias_array": [0.0],
    },
}

GODMAX_PARAMS = {
    "H0": 67.2,
    "Omega_cdm0": 0.261,
    "Omega_b0": 0.049,
    "sigma8": 0.81,
    "ns": 0.95,
    "w0": -1.0,
    "Omega_k0": 0.0,
    "wa": 0.0,
    "AIA": 0.3,
    "EtaIA": 0.1,
    "CIA": 0.0134,
    "multiplicative_bias_1": 0.01,
    "multiplicative_bias_2": 0.02,
    "dz_shear_1": 0.0,
    "dz_shear_2": 0.0,
    "theta_ej_0": 1.5,
}


@pytest.fixture(scope="module")
def godmax_available():
    pytest.importorskip("godmax.get_Cls")


def test_godmax_parameter_translation(godmax_available):
    theory = GODMAXPhotoTheory(GODMAX_CONFIG)
    theory.setup(make_data(("GGL",), "decoupled"), {"SHE", "POS"})

    params = dict(
        GODMAX_PARAMS,
        dz_shear_1=0.02,
        dz_shear_2=-0.01,
        multiplicative_bias_1=0.03,
        multiplicative_bias_2=0.04,
        b1_photo_poly0=1.0,
    )
    with pytest.warns(UserWarning, match="ignores parameters"):
        sim, halo, analysis, other = theory.godmax_inputs(params)

    assert sim["cosmo"]["Om0"] == pytest.approx(0.31)
    assert sim["cosmo"]["Ob0"] == pytest.approx(0.049)
    assert sim["cosmo"]["sigma8"] == pytest.approx(0.81)
    assert sim["theta_ej_0"] == pytest.approx(1.5)
    assert other["Delta_z_bias_array"] == pytest.approx([-0.02, 0.01])
    assert other["mult_shear_bias_array"] == pytest.approx([0.03, 0.04])
    assert (other["A_IA"], other["eta_IA"], other["z0_IA"]) == (0.3, 0.1, 0.0)
    assert other["C1_rhocrit"] == pytest.approx(0.0134)
    assert analysis["model_galaxies"] and analysis["model_tSZ"]
    assert analysis["is_cmb_lensing"] is False
    assert analysis["nz_source_info_dict"]["nbins"] == 2
    assert analysis["nz_lens_info_dict"]["nbins_lens"] == 2
    assert GODMAX_CONFIG["other_params"]["A_IA"] == 0.1  # config left untouched

    no_sigma8 = {k: v for k, v in GODMAX_PARAMS.items() if k != "sigma8"}
    with pytest.raises(ValueError, match="sigma8"):
        theory.godmax_inputs(dict(no_sigma8, As=2.1e-9))
    with pytest.raises(ValueError, match="flat"):
        theory.godmax_inputs(dict(GODMAX_PARAMS, Omega_k0=0.01))


@pytest.fixture(scope="module")
def godmax_all_probes(godmax_available):
    probes = tuple(PROBES)
    theory = GODMAXPhotoTheory(GODMAX_CONFIG)
    like = EuclidLikelihood_Cls(
        make_data(probes, "decoupled"),
        make_settings(probes),
        theory,
        probes=probes,
        mode="decoupled",
    )
    vector = like.get_theory_vector_full(GODMAX_PARAMS)
    return like, theory, vector


def test_godmax_limber_path_matches_native_spectra(godmax_all_probes):
    _, theory, _ = godmax_all_probes
    n_before = theory.n_evaluations
    godmax_cl = theory.evaluate(GODMAX_PARAMS)
    for key in [
        ("SHE", "SHE", 1, 2),
        ("POS", "POS", 2, 2),
        ("POS", "SHE", 2, 1),
        ("SHE", "Y", 1, 1),
        ("POS", "Y", 2, 1),
    ]:
        np.testing.assert_allclose(
            theory.limber_cell(godmax_cl, key),
            theory.native_cell(godmax_cl, key),
            rtol=1e-10,
        )
    assert theory.n_evaluations == n_before


def test_godmax_all_probes_likelihood(godmax_all_probes):
    like, theory, vector = godmax_all_probes
    assert vector.shape == (len(like.keys) * len(ELLS),)
    assert np.all(np.isfinite(vector))
    assert theory.n_evaluations == 1  # one GODMAX run shared by all ten probes
    for key in [("CMBL", "CMBL", 1, 1), ("Y", "Y", 1, 1), ("SHE", "SHE", 2, 2)]:
        assert np.all(e_mode_slice(key, like.theory_prediction[key]) > 0)

    # Data = fiducial theory: chi^2 vanishes there and grows when sigma8 moves.
    for key in like.keys:
        like.data["cells"][key] = like.theory_prediction[key]
    like.data["cov"] = np.diag((0.1 * vector) ** 2)
    assert like.loglike(GODMAX_PARAMS) == pytest.approx(0.0, abs=1e-10)
    assert theory.n_evaluations == 1
    assert like.loglike(dict(GODMAX_PARAMS, sigma8=0.85)) < -1.0
    assert theory.n_evaluations == 2


def test_godmax_coupled_mode(godmax_available):
    probes = ("WL", "CMBLxGC", "YY")
    like = EuclidLikelihood_Cls(
        make_data(probes, "coupled"),
        make_settings(probes),
        GODMAXPhotoTheory(GODMAX_CONFIG),
        probes=probes,
        mode="coupled",
    )
    vector = like.get_theory_vector_full(GODMAX_PARAMS)
    assert vector.shape == (len(like.keys) * len(ELLS),)
    assert np.all(np.isfinite(vector))
