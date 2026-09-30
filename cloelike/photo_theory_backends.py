"""Theory backends for the photometric angular power spectrum likelihoods.

A backend turns a parameter dictionary into *unmixed* angular power spectra for a
list of probe keys. The output follows the array conventions of
``cloelib.summary_statistics.angular_two_point.AngularTwoPoint.get_Cl``, so the
likelihood pipeline (mixing matrix, masking, data vector, covariance) does not
depend on which backend produced them:

* two shear legs, e.g. ``("SHE", "SHE", i, j)``: shape ``(2, 2, n_ell)``, E-mode in ``[0, 0]``
* one shear leg, e.g. ``("POS", "SHE", i, j)``: shape ``(2, n_ell)``, E-mode in ``[0]``
* no shear leg, e.g. ``("POS", "POS", i, j)``: shape ``(n_ell,)``

Fields: ``"SHE"`` cosmic shear, ``"POS"`` galaxy positions, ``"CMBL"`` CMB lensing
convergence (single map, bin index 1), ``"Y"`` thermal SZ Compton-y (single map,
bin index 1).

Backends
--------
``CloelibPhotoTheory``
    The theory path used by ``EuclidLikelihood_photo_Cls``: cloelib Background,
    linear/non-linear perturbations, tracers and Limber integration.
    Fields: SHE, POS, CMBL.
``GODMAXPhotoTheory``
    The GODMAX halo model. Fields: SHE, POS, CMBL, Y.

Every backend implements::

    supported_fields: frozenset
    derived: dict
    setup(data, fields) -> None
    get_cells(parameters, keys, ells) -> dict

``keys`` passed to ``get_cells`` all share the same field pair.
"""

import copy
import warnings
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from cloelib.observables.cmb import CMBLensingTracer
from cloelib.observables.photo import PositionsTracer, ShearTracer
from cloelib.summary_statistics.angular_two_point import AngularTwoPoint

SPIN2_FIELDS = frozenset({"SHE"})
ALL_FIELDS = frozenset({"SHE", "POS", "CMBL", "Y"})

# Same parameters, in the same order, as PhotoLikelihoodBase._BACKGROUND_PARAM_KEYS.
BACKGROUND_PARAM_KEYS = (
    "H0",
    "Omega_cdm0",
    "Omega_b0",
    "Omega_k0",
    "w0",
    "wa",
    "ns",
    "As",
    "mnu",
    "gamma_MG",
    "N_mnu",
    "alpha_s",
)

SPEED_OF_LIGHT_KM_S = 299792.458


def n_spin2_legs(key):
    """Number of spin-2 (shear) fields in a probe key."""
    return sum(field in SPIN2_FIELDS for field in key[:2])


def spin2_prefactor(ells):
    r"""Per-shear-leg Limber prefactor used by cloelib,
    :math:`\sqrt{(\ell+2)(\ell+1)\ell(\ell-1)} / (\ell+1/2)^2`."""
    ells = np.asarray(ells, dtype=float)
    return (
        np.sqrt(np.clip((ells + 2.0) * (ells + 1.0) * ells * (ells - 1.0), 0.0, None))
        / (ells + 0.5) ** 2
    )


def e_mode_slice(key, cl):
    """E-mode part of a C_ell array in cloelib layout (or the array itself for spin-0)."""
    arr = np.asarray(getattr(cl, "array", cl))
    legs = n_spin2_legs(key)
    if legs == 2 and arr.ndim == 3:
        return arr[0, 0]
    if legs == 1 and arr.ndim == 2:
        return arr[0]
    return arr


def embed_e_mode(key, cl):
    """Place an E-mode C_ell in the cloelib array layout for ``key`` (B-modes zero)."""
    cl = np.asarray(cl, dtype=float)
    legs = n_spin2_legs(key)
    if legs == 2:
        out = np.zeros((2, 2, cl.shape[-1]))
        out[0, 0] = cl
        return out
    if legs == 1:
        out = np.zeros((2, cl.shape[-1]))
        out[0] = cl
        return out
    return cl


def _freeze(parameters, names):
    """Hashable snapshot of ``parameters[name]`` for ``name`` in ``names``."""
    frozen = []
    for name in names:
        value = parameters[name]
        try:
            arr = np.asarray(value, dtype=float)
            frozen.append((name, arr.shape, arr.tobytes()))
        except (TypeError, ValueError):
            frozen.append((name, repr(value)))
    return tuple(frozen)


def _check_fields(backend, fields):
    unsupported = set(fields) - backend.supported_fields
    if unsupported:
        raise ValueError(
            f"{type(backend).__name__} cannot compute fields {sorted(unsupported)}; "
            f"supported fields are {sorted(backend.supported_fields)}."
        )


class CloelibPhotoTheory:
    """cloelib theory backend.

    Reproduces the theory computation of ``EuclidLikelihood_photo_Cls``:
    ``Background -> LinPerturbations -> NonLinPerturbations`` (cached on the
    cosmological parameters), ``ShearTracer`` / ``PositionsTracer`` /
    ``CMBLensingTracer`` (cached on their nuisance parameters), then
    ``AngularTwoPoint.get_Cl``.

    Parameters
    ----------
    Background, LinPerturbations, NonLinPerturbations : type
        Protocol-consistent cloelib classes, as passed to the existing likelihoods.
    nonlinear_parameter_keys : tuple of str
        Parameters forwarded as keyword arguments to ``NonLinPerturbations``.
    galaxy_bias_model : str
        Passed to ``PositionsTracer``.
    include_rsd : bool
        Passed to ``PositionsTracer``; the counterpart of the
        ``settings["include_rsd"]`` option of the existing likelihoods.
    """

    supported_fields = frozenset({"SHE", "POS", "CMBL"})

    def __init__(
        self,
        Background,
        LinPerturbations,
        NonLinPerturbations,
        nonlinear_parameter_keys=("log10TAGN",),
        galaxy_bias_model="poly",
        include_rsd=False,
    ):
        self.Background = Background
        self.LinPerturbations = LinPerturbations
        self.NonLinPerturbations = NonLinPerturbations
        self.nonlinear_parameter_keys = tuple(nonlinear_parameter_keys)
        self.galaxy_bias_model = galaxy_bias_model
        self.include_rsd = include_rsd
        self.derived = {}
        self.n_bins = {}

    def setup(self, data, fields):
        _check_fields(self, fields)
        self.zs = data["z_arr"]
        self._dndz = {}
        self.n_bins = {"CMBL": 1}
        if "SHE" in fields:
            self._dndz["SHE"] = data["dndz_she"]
            self.n_bins["SHE"] = data["dndz_she"].shape[0]
        if "POS" in fields:
            self._dndz["POS"] = data["dndz_pos"]
            self.n_bins["POS"] = data["dndz_pos"].shape[0]
        self._perturbations_cache = None
        self._tracer_cache = {}

    def nuisance_keys(self, field):
        """Nuisance parameter names consumed by the tracer of ``field``."""
        bins = range(1, self.n_bins.get(field, 0) + 1)
        if field == "SHE":
            return (
                ["AIA", "EtaIA", "CIA"]
                + [f"multiplicative_bias_{i}" for i in bins]
                + [f"dz_shear_{i}" for i in bins]
                + [f"width_shear_{i}" for i in bins]
            )
        if field == "POS":
            return (
                [f"b1_photo_poly{i}" for i in range(4)]
                + [f"magnification_bias_{i}" for i in bins]
                + [f"dz_pos_{i}" for i in bins]
                + [f"width_pos_{i}" for i in bins]
            )
        return []

    def _perturbations(self, parameters):
        key = _freeze(parameters, BACKGROUND_PARAM_KEYS + self.nonlinear_parameter_keys)
        if (
            self._perturbations_cache is not None
            and self._perturbations_cache[0] == key
        ):
            return self._perturbations_cache[1]
        background = self.Background(
            **{k: parameters[k] for k in BACKGROUND_PARAM_KEYS}
        )
        lp = self.LinPerturbations(background, self.zs)
        nlp = self.NonLinPerturbations(
            background,
            lp,
            self.zs,
            **{k: parameters[k] for k in self.nonlinear_parameter_keys},
        )
        # sigma8_0() memoizes by replacing itself on first call: call it exactly once.
        self.derived = {"sigma8_0": nlp.sigma8_0()}
        self._perturbations_cache = (key, nlp)
        self._tracer_cache = {}
        return nlp

    def _tracer(self, field, parameters, nlp):
        names = self.nuisance_keys(field)
        key = _freeze(parameters, names)
        cached = self._tracer_cache.get(field)
        if cached is not None and cached[0] == key:
            return cached[1]
        nuisance = {k: parameters[k] for k in names}
        if field == "SHE":
            tracer = ShearTracer(
                nlp, self._dndz["SHE"], self.zs, nuisance_params=nuisance
            )
        elif field == "POS":
            tracer = PositionsTracer(
                nlp,
                self._dndz["POS"],
                self.zs,
                nuisance_params=nuisance,
                galaxy_bias_model=self.galaxy_bias_model,
                include_rsd=self.include_rsd,
            )
        else:
            tracer = CMBLensingTracer(nlp, self.zs)
        self._tracer_cache[field] = (key, tracer)
        return tracer

    def get_cells(self, parameters, keys, ells):
        field1, field2 = keys[0][:2]
        nlp = self._perturbations(parameters)
        tracer1 = self._tracer(field1, parameters, nlp)
        tracer2 = tracer1 if field2 == field1 else self._tracer(field2, parameters, nlp)
        cells = AngularTwoPoint(tracer1, tracer2).get_Cl(ells, 0, nlp.k)
        return {key: np.asarray(cells[key].array) for key in keys}


class GODMAXPhotoTheory:
    """GODMAX halo-model theory backend.

    One GODMAX ``get_Cl`` object is built per parameter set and shared by every
    probe of the likelihood. Spectra are taken from GODMAX on its own ell grid
    (``halo_params`` ``ellmin``/``ellmax``/``nell``), interpolated with a cubic
    spline in ln(ell) onto the requested multipoles, and given the cloelib
    spin-2 prefactor on every shear leg.

    * SHE/POS/Y auto- and cross-spectra: GODMAX's ``Cl_*_tot_mat``; yy is the
      signal only (``Cl_y_y_signal_mat``; older GODMAX: ``Cl_y_y_tot_mat``).
    * Spectra with a CMBL leg: Limber integral of GODMAX's cached (ell, z) power
      spectra with the CMB lensing kernel, using the same integrand as GODMAX's
      ``get_Cl_tot``. GODMAX's own ``is_cmb_lensing`` switch replaces the shear
      kernel, so it is kept off and the CMB kernel is built here instead.

    Parameters
    ----------
    config : dict or str or pathlib.Path
        GODMAX parameters with sections ``sim_params``, ``halo_params``,
        ``analysis``, ``other_params`` (the layout of GODMAX's
        ``param_files/params_default.yaml``), or a path to such a YAML file.
        ``model_galaxies``, ``model_tSZ``, ``is_cmb_lensing`` and the n(z)
        dictionaries are set by this backend.
    parameter_map : dict, optional
        Extra ``{parameter_name: (section, godmax_key)}`` routes, with section
        one of ``"cosmo"``, ``"sim_params"``, ``"other_params"``. Applied last.

    Parameter translation
    ---------------------
    ========================  =========================================
    likelihood parameter      GODMAX input
    ========================  =========================================
    H0, ns, w0, sigma8        sim_params.cosmo
    Omega_cdm0 + Omega_b0     sim_params.cosmo.Om0 (and Ob0)
    AIA, EtaIA, CIA           other_params A_IA, eta_IA, C1_rhocrit, z0_IA=0
    multiplicative_bias_i     other_params.mult_shear_bias_array
    dz_shear_i                other_params.Delta_z_bias_array = -dz_shear_i
    any GODMAX config key     same key (e.g. theta_ej_0, log10M1_fshmr)
    ========================  =========================================

    GODMAX is flat w0CDM normalised by sigma8: ``As`` without ``sigma8``, a
    non-zero ``Omega_k0`` or a non-zero ``wa`` raise. Other parameters GODMAX
    cannot use (e.g. ``b1_photo_poly*``, ``magnification_bias_*``, ``width_*``,
    ``dz_pos_*``, ``log10TAGN``, ``mnu``) are ignored with a one-time warning.

    Notes
    -----
    Each new parameter set rebuilds the GODMAX object, which recompiles its
    jitted methods: expect seconds per evaluation even on small grids.
    """

    supported_fields = ALL_FIELDS
    _SECTIONS = ("sim_params", "halo_params", "analysis", "other_params")
    # Index of each field in GODMAX's cached_power_spectra (0 = baryonified matter).
    _PK_INDEX = {"SHE": 0, "CMBL": 0, "POS": 2, "Y": 3}

    def __init__(self, config, parameter_map=None):
        if isinstance(config, (str, Path)):
            import yaml

            with open(config) as f:
                config = yaml.safe_load(f)
        missing = [s for s in self._SECTIONS if s not in config]
        if missing:
            raise ValueError(f"GODMAX config is missing sections {missing}.")
        self._config = copy.deepcopy(config)
        self.parameter_map = dict(parameter_map or {})
        for name, route in self.parameter_map.items():
            if route[0] not in ("cosmo", "sim_params", "other_params"):
                raise ValueError(
                    f"parameter_map[{name!r}] section must be 'cosmo', "
                    f"'sim_params' or 'other_params', got {route[0]!r}."
                )

        try:
            import jax
            from godmax.get_Cls import get_Cl
        except ImportError as err:
            raise ImportError(
                "GODMAXPhotoTheory requires the 'godmax' package and its JAX stack."
            ) from err
        # GODMAX's halo-model integrals need double precision.
        jax.config.update("jax_enable_x64", True)
        self._get_Cl = get_Cl

        self.derived = {}
        self.n_evaluations = 0
        self.n_bins = {}
        self._warned = set()
        self._base = None
        self._cache = None

    def _warn_once(self, tag, message):
        if tag not in self._warned:
            self._warned.add(tag)
            warnings.warn(message, stacklevel=3)

    def setup(self, data, fields):
        _check_fields(self, fields)
        fields = set(fields)
        sim, halo, analysis, other = (
            copy.deepcopy(self._config[s]) for s in self._SECTIONS
        )
        analysis["model_galaxies"] = "POS" in fields
        # GODMAX's galaxy spectra use attributes created in its tSZ branch, so
        # galaxies need model_tSZ=True even when no y field is requested.
        analysis["model_tSZ"] = bool(fields & {"POS", "Y"})
        analysis["is_cmb_lensing"] = False

        z = np.asarray(data["z_arr"], dtype=float)
        self.n_bins = {"CMBL": 1, "Y": 1}
        if "SHE" in fields:
            dndz = np.asarray(data["dndz_she"], dtype=float)
            self.n_bins["SHE"] = dndz.shape[0]
            analysis["nz_source_info_dict"] = {
                "nbins": dndz.shape[0],
                "z_array_source": z,
                **{f"nz{i}": row for i, row in enumerate(dndz)},
            }
        if "POS" in fields:
            dndz = np.asarray(data["dndz_pos"], dtype=float)
            self.n_bins["POS"] = dndz.shape[0]
            analysis["nz_lens_info_dict"] = {
                "nbins_lens": dndz.shape[0],
                "z_array_lens": z,
                **{f"nz{i}": row for i, row in enumerate(dndz)},
            }
        self._check_redshift_coverage(data, fields, z, halo, analysis)

        self._base = (sim, halo, analysis, other)
        self._cache = None

    def _check_redshift_coverage(self, data, fields, z, halo, analysis):
        if "z_array" in halo:
            zmax_halo = float(np.max(halo["z_array"]))
        else:
            zmax_halo = float(halo.get("zmax", 1.5))
        zmax = min(float(analysis.get("zmax_for_Cls", 1.6)), zmax_halo)
        for field, name in (("SHE", "dndz_she"), ("POS", "dndz_pos")):
            if field not in fields:
                continue
            dndz = np.asarray(data[name], dtype=float)
            total = np.trapezoid(dndz, z, axis=-1)
            above = np.trapezoid(dndz * (z > zmax), z, axis=-1)
            frac = float(np.max(above / total))
            if frac > 0.01:
                warnings.warn(
                    f"{frac:.1%} of data['{name}'] lies above z = {zmax} (GODMAX "
                    "zmax_for_Cls / halo zmax); those redshifts are missing from the "
                    "GODMAX projection. Raise zmax_for_Cls and halo_params zmax.",
                    stacklevel=3,
                )

    def godmax_inputs(self, parameters):
        """Translate likelihood parameters into GODMAX's four input dictionaries."""
        if self._base is None:
            raise RuntimeError(
                "GODMAXPhotoTheory.setup(data, fields) has not been called."
            )
        sim, halo, analysis, other = (copy.deepcopy(s) for s in self._base)
        cosmo = sim.setdefault("cosmo", {})
        p = parameters
        used = set()

        for name in ("H0", "ns", "w0"):
            if name in p:
                cosmo[name] = p[name]
                used.add(name)
        if ("Omega_cdm0" in p) != ("Omega_b0" in p):
            raise ValueError(
                "GODMAX needs both 'Omega_cdm0' and 'Omega_b0' to set Om0."
            )
        if "Omega_cdm0" in p:
            cosmo["Om0"] = p["Omega_cdm0"] + p["Omega_b0"]
            cosmo["Ob0"] = p["Omega_b0"]
            used.update(("Omega_cdm0", "Omega_b0"))
        if "sigma8" in p:
            cosmo["sigma8"] = p["sigma8"]
            used.add("sigma8")
        elif "As" in p:
            raise ValueError(
                "GODMAX is normalised by sigma8 (jax_cosmo); pass 'sigma8' in the "
                "parameters instead of relying on 'As'."
            )
        for name in ("Omega_k0", "wa"):
            if name in p:
                used.add(name)
                if float(p[name]) != 0.0:
                    raise ValueError(
                        f"GODMAX assumes flat w0CDM; got {name} = {p[name]}."
                    )

        if "AIA" in p:
            other["A_IA"] = p["AIA"]
            used.add("AIA")
        if "EtaIA" in p:
            # cloelib's NLA redshift dependence is (1+z)^EtaIA, i.e. pivot z0 = 0.
            other["eta_IA"] = p["EtaIA"]
            other["z0_IA"] = 0.0
            used.add("EtaIA")
        if "CIA" in p:
            other["C1_rhocrit"] = p["CIA"]
            used.add("CIA")

        n_she = max(self.n_bins.get("SHE", 1), 1)
        for prefix, target, sign in (
            ("multiplicative_bias_", "mult_shear_bias_array", 1.0),
            # cloelib shifts n(z) -> n(z + dz); GODMAX shifts n(z) -> n(z - Delta_z).
            ("dz_shear_", "Delta_z_bias_array", -1.0),
        ):
            names = [f"{prefix}{i}" for i in range(1, n_she + 1)]
            present = [n for n in names if n in p]
            if present and "SHE" in self.n_bins:
                if len(present) != len(names):
                    raise ValueError(
                        f"Give all of {names} or none of them (got {present})."
                    )
                other[target] = [sign * float(p[n]) for n in names]
                used.update(names)
            values = np.atleast_1d(np.asarray(other.get(target, [0.0]), dtype=float))
            if values.size != n_she:
                if np.all(values == 0.0):
                    other[target] = [0.0] * n_she
                else:
                    raise ValueError(
                        f"other_params['{target}'] has {values.size} entries but "
                        f"there are {n_she} shear bins."
                    )

        ignored = []
        for name, value in p.items():
            if name in used:
                continue
            if name in cosmo and name != "flat":
                cosmo[name] = value
            elif name in sim and name != "cosmo":
                sim[name] = value
            elif name in other:
                other[name] = value
            elif name not in self.parameter_map:
                ignored.append(name)

        sections = {"cosmo": cosmo, "sim_params": sim, "other_params": other}
        for name, (section, key) in self.parameter_map.items():
            if name in p:
                sections[section][key] = p[name]

        if ignored:
            self._warn_once(
                ("ignored", tuple(sorted(ignored))),
                "GODMAXPhotoTheory ignores parameters it cannot model: "
                f"{sorted(ignored)}.",
            )
        return sim, halo, analysis, other

    def evaluate(self, parameters):
        """GODMAX ``get_Cl`` object for ``parameters`` (cached for the last parameter set)."""
        key = _freeze(parameters, sorted(parameters))
        if self._cache is not None and self._cache[0] == key:
            return self._cache[1]
        godmax_cl = self._get_Cl(*self.godmax_inputs(parameters))
        self.n_evaluations += 1
        self.derived = {"sigma8_0": float(godmax_cl.cosmo_params["sigma8"])}
        self._cache = (key, godmax_cl)
        return godmax_cl

    def native_cell(self, godmax_cl, key):
        """C_ell of ``key`` on the GODMAX ell grid, from GODMAX's own C_ell matrices."""
        field1, field2, i, j = key
        pair = (field1, field2)
        if pair == ("SHE", "SHE"):
            return np.asarray(godmax_cl.Cl_kappa_kappa_tot_mat[:, i - 1, j - 1])
        if pair == ("POS", "POS"):
            return np.asarray(godmax_cl.Cl_gal_gal_tot_mat[:, i - 1, j - 1])
        if pair == ("POS", "SHE"):
            return np.asarray(godmax_cl.Cl_gal_kappa_tot_mat[:, i - 1, j - 1])
        if pair == ("SHE", "Y"):
            return np.asarray(godmax_cl.Cl_kappa_y_tot_mat[:, i - 1])
        if pair == ("POS", "Y"):
            return np.asarray(godmax_cl.Cl_gal_y_tot_mat[:, i - 1])
        if pair == ("Y", "Y"):
            signal = getattr(godmax_cl, "Cl_y_y_signal_mat", None)
            if signal is None:
                signal = godmax_cl.Cl_y_y_tot_mat
            return np.asarray(signal)
        raise ValueError(f"No GODMAX C_ell matrix for key {key}.")

    def cmb_lensing_kernel(self, godmax_cl):
        """CMB lensing kernel on GODMAX's z grid, as GODMAX's get_cmb_lensing_kernel."""
        z = np.asarray(godmax_cl.z_array_for_Cls)
        chi = np.asarray(godmax_cl.chi_array_for_Cls)
        chi_star = float(np.ravel(np.asarray(godmax_cl.chi_CMB))[0])
        amplitude = (
            3.0
            * 100.0**2
            * float(godmax_cl.cosmo_jax.Omega_m)
            / (2.0 * SPEED_OF_LIGHT_KM_S**2)
        )
        return (
            amplitude * np.clip(chi_star - chi, 0.0, None) / chi_star * (1.0 + z) * chi
        )

    def _kernel_prefactor(self, godmax_cl, field, jb):
        chi = np.asarray(godmax_cl.chi_array_for_Cls)
        if field == "SHE":
            m = float(godmax_cl.mult_shear_bias_array[jb])
            return (1.0 + m) * np.asarray(godmax_cl.Wk_mat[jb]) / chi**2
        if field == "POS":
            dchi_dz = np.asarray(godmax_cl.dchi_dz_array_for_Cls)
            return np.asarray(godmax_cl.Wg_mat[jb]) / (dchi_dz * chi**2)
        if field == "Y":
            return np.asarray(godmax_cl.Wy_array) / chi**2
        if field == "CMBL":
            return self.cmb_lensing_kernel(godmax_cl) / chi**2
        raise ValueError(f"Unknown field {field!r}.")

    def limber_cell(self, godmax_cl, key):
        """C_ell of ``key`` on the GODMAX ell grid from the Limber integral of GODMAX's
        cached (ell, z) power spectra; the same integrand as GODMAX's get_Cl_tot."""
        field1, field2, i, j = key
        z = np.asarray(godmax_cl.z_array_for_Cls)
        chi = np.asarray(godmax_cl.chi_array_for_Cls)
        dchi_dz = np.asarray(godmax_cl.dchi_dz_array_for_Cls)
        weight = (
            self._kernel_prefactor(godmax_cl, field1, i - 1)
            * self._kernel_prefactor(godmax_cl, field2, j - 1)
            * chi**2
            * dchi_dz
        )
        pk = np.asarray(
            godmax_cl.cached_power_spectra[
                self._PK_INDEX[field1], self._PK_INDEX[field2]
            ]
        )
        return np.trapezoid(weight[None, :] * pk, x=z, axis=-1)

    def grid_cell(self, godmax_cl, key):
        if "CMBL" in key[:2]:
            return self.limber_cell(godmax_cl, key)
        return self.native_cell(godmax_cl, key)

    def get_cells(self, parameters, keys, ells):
        godmax_cl = self.evaluate(parameters)
        ell_grid = np.asarray(godmax_cl.ell_array, dtype=float)
        ells = np.asarray(ells, dtype=float)

        ell_used = ells[ells >= 2.0] if np.any(ells >= 2.0) else ells
        if ell_used.min() < ell_grid[0] * (1 - 1e-8) or ell_used.max() > ell_grid[
            -1
        ] * (1 + 1e-8):
            self._warn_once(
                ("ell_range", float(ell_used.min()), float(ell_used.max())),
                f"C_ell requested for ell in [{ell_used.min():g}, {ell_used.max():g}] "
                f"but the GODMAX ell grid spans [{ell_grid[0]:g}, {ell_grid[-1]:g}]; "
                "values are held constant outside the grid. Adjust halo_params "
                "ellmin/ellmax to cover the requested range.",
            )

        grid = np.stack([self.grid_cell(godmax_cl, key) for key in keys])
        spline = CubicSpline(np.log(ell_grid), grid, axis=-1)
        values = spline(np.log(np.clip(ells, ell_grid[0], ell_grid[-1])))
        prefactor = spin2_prefactor(ells)
        return {
            key: embed_e_mode(key, value * prefactor ** n_spin2_legs(key))
            for key, value in zip(keys, values)
        }
