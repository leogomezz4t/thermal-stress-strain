# thermal_stress_v3.py
#
# Thermoelastic model of the piezo stack under uniform cooling to cryogenic
# temperature.  Temperature is uniform (T_CRYO everywhere), so no heat equation
# is needed.  Stress arises from differential thermal contraction between the
# four materials: LiNbO3 layers, aluminum electrodes, Macor caps, 353ND epoxy.
#
# Improvements over v2
# --------------------
# 1. INTEGRATED THERMAL STRAIN.
#    v2 used a constant coefficient of thermal expansion:  ε_th = α·ΔT.
#    But α varies enormously between 293 K and 4 K — it falls roughly as T³
#    below the Debye temperature and is essentially zero below ~20 K.
#    Using α(4K)·ΔT underestimates the contraction by orders of magnitude;
#    using α(293K)·ΔT overestimates it by ~40%.  The correct eigenstrain is
#
#        ε_th = ∫_{T_REF}^{T_CRYO} α(T) dT      (= ΔL/L of free contraction)
#
#    Here each material carries a tabulated α(T) curve and the integral is
#    evaluated numerically (trapezoid rule).  This also makes it trivial to
#    re-run the model at intermediate temperatures (e.g. 77 K): just change
#    T_CRYO — the integral picks up the right partial contraction.
#
# 2. ANISOTROPIC LiNbO3.
#    v2 treated LiNbO3 as isotropic.  LiNbO3 is a trigonal crystal (class 3m)
#    with 6 independent elastic constants and a strongly direction-dependent
#    thermal expansion (α_a ≈ 2× α_c).  v3 stores the full 6×6 Voigt stiffness
#    matrix per cell and a 6-component Voigt thermal-strain vector per cell,
#    so isotropic and anisotropic materials are handled by one code path.
#    The crystal cut (orientation of the crystal axes relative to the stack)
#    is a single rotation matrix — see CRYSTAL_ROTATION below.
#
# 3. WELL-POSED BOUNDARY CONDITIONS.
#    v2 solved with no Dirichlet BCs at all (the clamps were commented out),
#    which leaves the 6 rigid-body modes in the null space — the LU solve
#    only "works" by floating-point accident.  v3 pins exactly 6 DOFs at three
#    corner vertices (the classic "3-2-1" scheme).  This is statically
#    determinate: it removes rigid translation/rotation but exerts no force,
#    so it adds zero spurious stress.  All stress comes from CTE mismatch.
#
# Conventions
# -----------
#   Mesh coordinates are in mm and moduli in MPa, so stress is in MPa (N/mm²).
#   Voigt order: (xx, yy, zz, yz, xz, xy) with ENGINEERING shear strain
#   γ = 2ε for the shear components (so C44 = μ for isotropic materials).

from petsc4py.PETSc import ScalarType  # type: ignore
import numpy as np
import ufl
import pyvista

from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.plot import vtk_mesh
from dolfinx.mesh import uniform_refine

from stacks.LN_Al_full import LN_Al_Full
from stacks.full_piezo_stack import PIEZO_TAG_EVEN, PIEZO_TAG_ODD, CAP_TAG, EPOXY_TAG, ELECTRODE_TAG
from materials import Materials

# ============================================================
# THERMAL LOADING
# ============================================================
T_REF  = 293.0   # Assembly temperature — the stress-free reference state [K]
T_CRYO = 4.0     # Cryogenic operating temperature [K]

# ============================================================
# VISUALISATION
# ============================================================
WARP_SCALE = 200   # Displacement magnification factor for the deformed mesh plot

# ============================================================
# BOUNDARY CONDITIONS
# False → free contraction, rigid-body modes pinned by 3-2-1 point constraints.
# True  → top and bottom faces fully clamped (u = 0), as if bonded to
#         infinitely stiff fixtures.  Reality is between the two extremes.
# ============================================================
CLAMP_ENDS = False

# ============================================================
# CRYSTAL ORIENTATION OF THE LiNbO3 LAYERS
#
# CRYSTAL_ROTATION maps crystal-frame components to mesh-frame components:
#     v_mesh = R @ v_crystal        (columns of R = crystal axes X,Y,Z
#                                    expressed in mesh coordinates)
# The crystal Z axis is the optic (c) axis.
#
# Default: Z-cut — crystal c-axis along the stack axis (mesh z).
# Other common cuts (replace R below):
#   X-cut  (crystal X along stack z):
#       R = np.array([[0,0,1],[1,0,0],[0,1,0]], dtype=float).T ... or simply
#       any R whose THIRD COLUMN is the mesh direction of the crystal Z axis.
#   Rotated Y-cut by angle θ about crystal X (e.g. 36°Y):
#       th = np.radians(36)
#       R = np.array([[1, 0,           0          ],
#                     [0, np.cos(th), -np.sin(th)],
#                     [0, np.sin(th),  np.cos(th)]])
# Both the stiffness tensor and the thermal-expansion tensor are rotated
# with this R, so changing the cut is a one-line edit.
# ============================================================
th = np.radians(36)
X_CUT_ODD = np.array([[0,0,1],[1,0,0],[0,1,0]], dtype=float).T   # X-cut
X_CUT_EVEN = np.array([[0,0,1], [-1, 0, 0], [0, -1, 0]], dtype=float).T
Y_36_CUT_ODD = np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]])
Y_36_CUT_EVEN = np.array([[1, 0, 0], [0, -np.cos(th), np.sin(th)], [0, -np.sin(th), -np.cos(th)]])



CRYSTAL_ROTATION     = X_CUT_EVEN   # rotation for PIEZO_TAG_ODD layers
CRYSTAL_ROTATION_ALT = X_CUT_ODD  # rotation for PIEZO_TAG_EVEN layers

# ============================================================
# TEMPERATURE-DEPENDENT THERMAL EXPANSION DATA
#
# Each entry is a table of the linear thermal expansion coefficient α(T)
# sampled from 4 K to 293 K.  α(T) → 0 as T → 0 for all solids (3rd law),
# which is why the constant-α model fails so badly across a cryogenic range.
#
# The thermal eigenstrain is the integral of these curves (see
# integrated_thermal_strain below).  Sanity anchor: NIST gives the total
# contraction ΔL/L (293 K → 4 K) of aluminum 6061 as ≈ −0.415%; the table
# below integrates to −0.417%.
#
# DOUBLE CHECK — curve shapes are AI-estimated from typical published data;
# verify against NIST cryogenic material properties / Ekin "Experimental
# Techniques for Low-Temperature Measurements" before trusting:
#   ALUMINUM     — medium-high confidence (NIST TRC 6061 shape, anchor above).
#   MACOR        — LOW.  Corning quotes α ≈ 9.3e-6 near RT; the low-T rolloff
#                  is assumed ceramic-like.  Integrated total ≈ −0.16%.
#   EPOXY_353ND  — VERY LOW.  Unfilled-epoxy-like curve, total ≈ −0.9%
#                  (cf. Stycast 1266 ≈ −1.15%).  Measure or ask Epo-Tek.
#   LiNbO3       — LOW.  RT anchors α_a ≈ 15.4e-6, α_c ≈ 7.5e-6 (Smith &
#                  Welsh 1971); low-T rolloff assumed Debye-like.  Some
#                  literature reports anomalous (possibly negative) α_c at
#                  low temperature — MUST verify before trusting c-axis strain.
# ============================================================

# Temperatures [K] shared by all tables, ascending
ALPHA_T = np.array([4.0, 20.0, 40.0, 60.0, 80.0, 100.0, 140.0, 180.0, 220.0, 260.0, 293.0])

ALPHA_TABLES = {
    #                       4K    20K   40K   60K   80K   100K  140K  180K  220K  260K  293K   [1e-6/K]
    "ALUMINUM":    np.array([0.0,  0.5,  2.6,  5.6,  8.9, 12.2, 16.6, 19.3, 21.0, 22.0, 22.7]) * 1e-6,
    "MACOR":       np.array([0.0,  0.2,  0.8,  1.8,  3.0,  4.2,  6.2,  7.6,  8.5,  9.1,  9.4]) * 1e-6,
    "EPOXY_353ND": np.array([0.5,  4.5, 10.0, 15.0, 19.5, 24.0, 31.2, 37.8, 44.2, 49.7, 54.0]) * 1e-6,
    "LINBO3_A":    np.array([0.0,  0.3,  1.5,  3.4,  5.5,  7.5, 10.7, 12.7, 14.0, 14.8, 15.4]) * 1e-6,  # ⊥ c-axis
    "LINBO3_C":    np.array([0.0,  0.1,  0.5,  1.2,  2.2,  3.2,  4.9,  6.0,  6.8,  7.3,  7.5]) * 1e-6,  # ∥ c-axis
}


def integrated_thermal_strain(alpha_table: np.ndarray, T_from: float, T_to: float) -> float:
    """ε_th = ∫_{T_from}^{T_to} α(T) dT, by trapezoid rule on a fine grid.

    Returns the free linear strain ΔL/L of the material between the two
    temperatures.  Negative for cooling (T_to < T_from): the material wants
    to shrink, and whatever shrinkage the surrounding structure prevents
    shows up as stress.

    The table is linearly interpolated onto a fine grid first so that
    T_from / T_to need not coincide with table points (e.g. T_CRYO = 77 K).
    """
    lo, hi = min(T_from, T_to), max(T_from, T_to)
    T_grid = np.linspace(lo, hi, 2000)
    alpha  = np.interp(T_grid, ALPHA_T, alpha_table)
    integral = float(np.sum(0.5 * (alpha[1:] + alpha[:-1]) * np.diff(T_grid)))
    return integral if T_to >= T_from else -integral


# ============================================================
# LiNbO3 ANISOTROPIC STIFFNESS (crystal frame)
#
# Trigonal class 3m → 6 independent constants.  Voigt matrix structure:
#
#        ⎡ c11  c12  c13  c14   0    0  ⎤
#        ⎢ c12  c11  c13 -c14   0    0  ⎥
#   C =  ⎢ c13  c13  c33   0    0    0  ⎥      c66 = (c11 − c12)/2
#        ⎢ c14 -c14   0   c44   0    0  ⎥
#        ⎢  0    0    0    0   c44  c14 ⎥
#        ⎣  0    0    0    0   c14  c66 ⎦
#
# Values: Warner, Onoe & Coquin (1967), constant-field constants c^E [GPa].
# The c14 term couples normal strain in the basal plane to shear — its SIGN
# depends on the choice of +X axis (IEEE 1978 convention used here); its
# magnitude is small, so an orientation sign error is a minor effect.
#
# These are room-temperature constants.  materials.py estimates ~3%
# stiffening at 4 K for LiNbO3 (by analogy with other oxide ceramics, LOW
# confidence) — applied uniformly here via CRYO_STIFFENING.
# ============================================================

CRYO_STIFFENING = Materials.LITHIUM_NIOBATE_4K.E / Materials.LITHIUM_NIOBATE.E   # ≈ +3%, matches the materials.py 4K estimate

_c11, _c12, _c13 = 203_000.0, 57_300.0, 75_200.0   # [MPa]
_c14, _c33, _c44 = 8_500.0, 242_400.0, 59_500.0    # [MPa]
_c66 = (_c11 - _c12) / 2

C_LINBO3_ROOM = np.array([
    [_c11,  _c12,  _c13,  _c14,  0.0,   0.0 ],
    [_c12,  _c11,  _c13, -_c14,  0.0,   0.0 ],
    [_c13,  _c13,  _c33,  0.0,   0.0,   0.0 ],
    [_c14, -_c14,  0.0,   _c44,  0.0,   0.0 ],
    [0.0,   0.0,   0.0,   0.0,   _c44,  _c14],
    [0.0,   0.0,   0.0,   0.0,   _c14,  _c66],
])

C_LINBO3_CRYSTAL = CRYO_STIFFENING * C_LINBO3_ROOM
# ============================================================
# VOIGT / TENSOR UTILITIES
#
# A 6×6 Voigt stiffness matrix cannot be rotated by R directly — rotation is
# defined on the underlying 4th-order tensor C_ijkl.  So: unpack 6×6 → 3×3×3×3,
# rotate with C'_ijkl = R_ia R_jb R_kc R_ld C_abcd, repack.  The engineering-
# shear convention makes the 6×6 ↔ 4th-order mapping factor-free for STIFFNESS
# (factors of 2 live in the strain vector instead).
# ============================================================

# Voigt index → tensor index pair, in our (xx, yy, zz, yz, xz, xy) order
VOIGT_PAIRS = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


def stiffness_voigt_to_tensor(C6: np.ndarray) -> np.ndarray:
    """6×6 Voigt stiffness → full 3×3×3×3 tensor (minor symmetries filled in)."""
    C4 = np.zeros((3, 3, 3, 3))
    for I, (i, j) in enumerate(VOIGT_PAIRS):
        for J, (k, l) in enumerate(VOIGT_PAIRS):
            C4[i, j, k, l] = C4[j, i, k, l] = C4[i, j, l, k] = C4[j, i, l, k] = C6[I, J]
    return C4


def stiffness_tensor_to_voigt(C4: np.ndarray) -> np.ndarray:
    """Full 3×3×3×3 stiffness tensor → 6×6 Voigt matrix."""
    C6 = np.zeros((6, 6))
    for I, (i, j) in enumerate(VOIGT_PAIRS):
        for J, (k, l) in enumerate(VOIGT_PAIRS):
            C6[I, J] = C4[i, j, k, l]
    return C6


def rotate_stiffness(C6: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Rotate a Voigt stiffness matrix from crystal frame to mesh frame."""
    C4 = stiffness_voigt_to_tensor(C6)
    C4_rot = np.einsum("ia,jb,kc,ld,abcd->ijkl", R, R, R, R, C4)
    return stiffness_tensor_to_voigt(C4_rot)


def strain_tensor_to_voigt(e: np.ndarray) -> np.ndarray:
    """Symmetric 3×3 strain tensor → 6-vector with engineering shear (γ = 2ε)."""
    return np.array([e[0, 0], e[1, 1], e[2, 2],
                     2 * e[1, 2], 2 * e[0, 2], 2 * e[0, 1]])


def isotropic_stiffness_voigt(E: float, nu: float) -> np.ndarray:
    """Isotropic 6×6 Voigt stiffness from Young's modulus and Poisson's ratio.

    λ = Eν / ((1+ν)(1−2ν)),  μ = E / (2(1+ν)).
    With engineering shear strain the shear diagonal is μ (not 2μ).
    """
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu  = E / (2 * (1 + nu))
    C6 = np.zeros((6, 6))
    C6[:3, :3] = lam
    C6[np.arange(3), np.arange(3)] = lam + 2 * mu
    C6[np.arange(3, 6), np.arange(3, 6)] = mu
    return C6


# ============================================================
# MESH
# ============================================================

piezo_mesh = LN_Al_Full()

MATERIAL_NAMES = {
    PIEZO_TAG_ODD:     "LiNbO3",
    PIEZO_TAG_EVEN: "LiNbO3",
    ELECTRODE_TAG: "Aluminum",
    CAP_TAG:       "Macor",
    EPOXY_TAG:     "Epoxy 353ND",
}

msh  = piezo_mesh.msh
tdim = piezo_mesh.tdim

# ============================================================
# PER-MATERIAL STIFFNESS AND THERMAL EIGENSTRAIN (mesh frame)
#
# For each material we compute, in mesh coordinates:
#   C6      — 6×6 Voigt stiffness matrix [MPa]
#   eps_th  — 6-component Voigt thermal eigenstrain (engineering shear)
#
# Isotropic materials: stiffness from the 4 K E, ν in materials.py (correct
# stiffness at the operating point), eigenstrain = ∫α dT × identity.
#
# LiNbO3: rotate the crystal-frame stiffness and the crystal-frame expansion
# tensor diag(∫α_a, ∫α_a, ∫α_c) into the mesh frame.  Off-diagonal eigenstrain
# components appear automatically for rotated cuts.
#
# Note on path dependence: rigorously, stress accumulates along the cooldown
# as dσ = C(T) d(ε − ε_th), and C itself changes a few percent (up to ~2–4×
# for the epoxy) between 293 K and 4 K.  Using the 4 K stiffness with the
# fully integrated eigenstrain is the standard one-step approximation; its
# error is bounded by the relative stiffness change and is small next to the
# uncertainty in the α(T) data itself.
# ============================================================

R = CRYSTAL_ROTATION

# Scalar integrated strains (negative — everything shrinks on cooling)
eth_iso = {
    name: integrated_thermal_strain(ALPHA_TABLES[name], T_REF, T_CRYO)
    for name in ("ALUMINUM", "MACOR", "EPOXY_353ND")
}
eth_a = integrated_thermal_strain(ALPHA_TABLES["LINBO3_A"], T_REF, T_CRYO)
eth_c = integrated_thermal_strain(ALPHA_TABLES["LINBO3_C"], T_REF, T_CRYO)

print(f"Integrated thermal strain {T_REF:.0f} K → {T_CRYO:.0f} K  (ΔL/L of free contraction):")
print(f"  Aluminum:      {eth_iso['ALUMINUM']*100:+.4f} %")
print(f"  Macor:         {eth_iso['MACOR']*100:+.4f} %")
print(f"  Epoxy 353ND:   {eth_iso['EPOXY_353ND']*100:+.4f} %")
print(f"  LiNbO3 a-axis: {eth_a*100:+.4f} %   c-axis: {eth_c*100:+.4f} %")


def linbo3_C6_and_eth(rot: np.ndarray):
    """Stiffness (6×6) and eigenstrain (6-vector) for LiNbO3 in a given orientation."""
    C6  = rotate_stiffness(C_LINBO3_CRYSTAL, rot)
    eth = strain_tensor_to_voigt(rot @ np.diag([eth_a, eth_a, eth_c]) @ rot.T)
    return C6, eth


material_C6 = {
    ELECTRODE_TAG: isotropic_stiffness_voigt(piezo_mesh.materials[ELECTRODE_TAG].E,
                                             piezo_mesh.materials[ELECTRODE_TAG].nu),
    CAP_TAG:       isotropic_stiffness_voigt(piezo_mesh.materials[CAP_TAG].E,
                                             piezo_mesh.materials[CAP_TAG].nu),
    EPOXY_TAG:     isotropic_stiffness_voigt(piezo_mesh.materials[EPOXY_TAG].E,
                                             piezo_mesh.materials[EPOXY_TAG].nu),
}

material_eps_th = {
    ELECTRODE_TAG: eth_iso["ALUMINUM"]    * np.array([1, 1, 1, 0, 0, 0], dtype=float),
    CAP_TAG:       eth_iso["MACOR"]       * np.array([1, 1, 1, 0, 0, 0], dtype=float),
    EPOXY_TAG:     eth_iso["EPOXY_353ND"] * np.array([1, 1, 1, 0, 0, 0], dtype=float),
}

# Store per-cell data in DG0 (piecewise-constant) spaces: a 6×6 matrix and a
# 6-vector on every cell.  DG0 is the natural home for material data — exact
# jumps at material interfaces, no smearing.
DG0_C   = fem.functionspace(msh, ("DG", 0, (6, 6)))
DG0_eth = fem.functionspace(msh, ("DG", 0, (6,)))

C_fn      = fem.Function(DG0_C)
eps_th_fn = fem.Function(DG0_eth)

C_flat   = C_fn.x.array.reshape(-1, 36)       # one flattened 6×6 per cell
eth_flat = eps_th_fn.x.array.reshape(-1, 6)   # one 6-vector per cell

# Assign isotropic materials.
for tag in (ELECTRODE_TAG, CAP_TAG, EPOXY_TAG):
    cells = piezo_mesh.cell_tags.find(tag)
    C_flat[cells, :]   = material_C6[tag].flatten()
    eth_flat[cells, :] = material_eps_th[tag]

# Assign LiNbO3 — even and odd layers carry different crystal orientations.
C6_odd,  eth6_odd  = linbo3_C6_and_eth(CRYSTAL_ROTATION)
C6_even, eth6_even = linbo3_C6_and_eth(CRYSTAL_ROTATION_ALT)

for tag, C6, eth6 in ((PIEZO_TAG_ODD,  C6_odd,  eth6_odd),
                      (PIEZO_TAG_EVEN, C6_even, eth6_even)):
    cells = piezo_mesh.cell_tags.find(tag)
    C_flat[cells, :]   = C6.flatten()
    eth_flat[cells, :] = eth6

# ============================================================
# WEAK FORM
#
# Static equilibrium with thermal eigenstrain:
#       −∇·σ = 0,        σ = C : (ε(u) − ε_th)
#
# Multiply by test function v, integrate by parts (traction-free boundaries
# kill the surface term), and split σ into its u-dependent and known parts:
#
#       ∫ [C : ε(u)] : ε(v) dx  =  ∫ [C : ε_th] : ε(v) dx
#       └────── a(u,v) ──────┘     └────── L(v) ───────┘
#
# The right-hand side is the "thermal pre-stress" C:ε_th tested against ε(v) —
# the anisotropic generalisation of v2's (3λ+2μ)·α·ΔT·div(v) term.
#
# Everything is computed in Voigt form (σ_v = C6 · ε_v) and converted back to
# a 3×3 tensor for the inner product, which keeps the UFL close to the math.
# ============================================================

V_u = fem.functionspace(msh, ("Lagrange", 1, (tdim,)))


def eps(u):
    """Symmetric strain tensor ε(u) = ½(∇u + ∇uᵀ)."""
    return ufl.sym(ufl.grad(u))


def eps_voigt(u):
    """Strain in Voigt form (xx, yy, zz, 2yz, 2xz, 2xy) — engineering shear."""
    e = eps(u)
    return ufl.as_vector([e[0, 0], e[1, 1], e[2, 2],
                          2 * e[1, 2], 2 * e[0, 2], 2 * e[0, 1]])


def voigt_to_tensor(s):
    """Voigt stress 6-vector back to a symmetric 3×3 tensor.

    (No factors of 2: those belong only on the STRAIN side of the
    engineering-shear convention.)
    """
    return ufl.as_tensor([[s[0], s[5], s[4]],
                          [s[5], s[1], s[3]],
                          [s[4], s[3], s[2]]])


def sigma_elastic(u):
    """Stress from displacement alone: σ_el = C : ε(u)."""
    return voigt_to_tensor(ufl.dot(C_fn, eps_voigt(u)))


def sigma_total(u):
    """Physical stress including the thermal eigenstrain: σ = C : (ε(u) − ε_th)."""
    return voigt_to_tensor(ufl.dot(C_fn, eps_voigt(u) - eps_th_fn))


u = ufl.TrialFunction(V_u)
v = ufl.TestFunction(V_u)

a_u = ufl.inner(sigma_elastic(u), eps(v)) * ufl.dx
L_u = ufl.inner(voigt_to_tensor(ufl.dot(C_fn, eps_th_fn)), eps(v)) * ufl.dx

# ============================================================
# BOUNDARY CONDITIONS
#
# A floating body has 6 rigid-body modes (3 translations + 3 rotations);
# pin exactly 6 displacement components at three bottom corner vertices:
#   A = (x_min, y_min, z_min): ux = uy = uz = 0   (kills translations)
#   B = (x_max, y_min, z_min): uy = uz = 0        (kills rotations about z, y)
#   C = (x_min, y_max, z_min): uz = 0             (kills rotation about x)
# Statically determinate → reaction forces are identically zero, so the
# constraints add NO stress.  (Check: for a homogeneous body contracting
# uniformly about A, u = ε_th·(x − x_A) satisfies all six constraints exactly.)
# The choice of anchor point shifts the displacement *picture* (everything
# moves toward A) but leaves strain and stress unchanged.
# ============================================================

geom = msh.geometry.x
x_min, y_min, z_min = geom.min(axis=0)
x_max, y_max, _     = geom.max(axis=0)
TOL = 1e-6  # [mm] — corner vertices sit at exact box coordinates


def _at(px, py, pz):
    """Marker selecting the single mesh vertex at (px, py, pz)."""
    return lambda x: (np.isclose(x[0], px, atol=TOL)
                      & np.isclose(x[1], py, atol=TOL)
                      & np.isclose(x[2], pz, atol=TOL))


def component_point_bc(component, marker):
    """Zero-Dirichlet BC on one displacement component at one point.

    Single components of a blocked vector space must go through the
    sub-space/collapse machinery; this hides the boilerplate.
    """
    sub = V_u.sub(component)
    collapsed, _ = sub.collapse()
    dofs = fem.locate_dofs_geometrical((sub, collapsed), marker)
    zero = fem.Function(collapsed)   # initialised to 0
    return fem.dirichletbc(zero, dofs, sub)


if CLAMP_ENDS:
    fdim = piezo_mesh.fdim
    bottom_dofs = fem.locate_dofs_topological(V_u, fdim, piezo_mesh.bottom_facets)
    top_dofs    = fem.locate_dofs_topological(V_u, fdim, piezo_mesh.top_facets)
    zero_vec = np.zeros(tdim, dtype=ScalarType)
    bcs_u = [
        fem.dirichletbc(zero_vec, bottom_dofs, V_u),
        fem.dirichletbc(zero_vec, top_dofs,    V_u),
    ]
else:
    dofs_A = fem.locate_dofs_geometrical(V_u, _at(x_min, y_min, z_min))
    bcs_u = [
        fem.dirichletbc(np.zeros(tdim, dtype=ScalarType), dofs_A, V_u),  # A: all 3
        component_point_bc(1, _at(x_max, y_min, z_min)),                 # B: uy
        component_point_bc(2, _at(x_max, y_min, z_min)),                 # B: uz
        component_point_bc(2, _at(x_min, y_max, z_min)),                 # C: uz
    ]

# ============================================================
# SOLVE
# ============================================================

stress_problem = LinearProblem(
    a_u, L_u, bcs=bcs_u,
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    petsc_options_prefix="stress_",
)
uh = stress_problem.solve()
uh.name = "u"

disp_mag = np.linalg.norm(uh.x.array.real.reshape(-1, tdim), axis=1)
print(f"Displacement — max magnitude: {disp_mag.max():.4e} mm")

# ============================================================
# POST-PROCESSING: Von Mises stress
# σ_VM = sqrt(3/2 · s:s)   where s = σ − (1/3) tr(σ) I  (deviatoric part)
# Von Mises is the yield criterion for the ductile aluminum electrodes.
# (For the anisotropic LiNbO3 crystal it is only a rough indicator — single
# crystals fail on specific cleavage planes — but it is kept for comparison.)
# ============================================================

W = fem.functionspace(msh, ("DG", 0))

sig     = sigma_total(uh)
s       = sig - (1 / 3) * ufl.tr(sig) * ufl.Identity(tdim)
vm_expr = ufl.sqrt(3 / 2 * ufl.inner(s, s))

von_mises = fem.Function(W)
von_mises.interpolate(fem.Expression(vm_expr, W.element.interpolation_points))
von_mises.name = "von_mises"

vm_array = von_mises.x.array.real
max_vm   = vm_array.max()
print(f"Von Mises stress — max: {max_vm:.2f} MPa")

# ============================================================
# POST-PROCESSING: Maximum principal stress (σ₁)
# LiNbO3 and Macor are brittle — they fracture in tension when σ₁ exceeds
# the tensile fracture strength.  Hydrostatic stress contributes zero to
# Von Mises but fully to σ₁, so the two criteria can disagree significantly.
# ============================================================

W_tensor = fem.functionspace(msh, ("DG", 0, (tdim, tdim)))
sig_fn   = fem.Function(W_tensor)
sig_fn.interpolate(fem.Expression(sigma_total(uh), W_tensor.element.interpolation_points))

# eigvalsh returns eigenvalues sorted ascending: σ₃ ≤ σ₂ ≤ σ₁
sig_cells     = sig_fn.x.array.real.reshape(-1, tdim, tdim)
principal     = np.linalg.eigvalsh(sig_cells)   # (n_cells, 3)
max_principal = principal[:, -1]                 # σ₁ per cell (most tensile)
max_sp        = float(max_principal.max())
print(f"Max principal stress σ₁: {max_sp:.2f} MPa")

# Fracture legend on the σ₁ panel — the correct comparison for brittle materials
fracture_legend = [
    [f"{MATERIAL_NAMES[tag]}  σ_f = {mat.sigma_f} MPa", "black"]
    for tag, mat in piezo_mesh.materials.items()
    if tag in MATERIAL_NAMES
]

# ============================================================
# VISUALISATION: three panels
#   Left   — displacement magnitude on warped mesh
#   Centre — Von Mises stress with clip plane through the peak
#   Right  — max principal stress with clip plane through the peak
# ============================================================

# Displacement grid (vector, point data) — warp for visibility
c_u, t_u, x_u = vtk_mesh(V_u)
grid_u = pyvista.UnstructuredGrid(c_u, t_u, x_u)
disp_vec = uh.x.array.real.reshape(-1, tdim)
grid_u.point_data["disp"]     = disp_vec
grid_u.point_data["disp_mag"] = np.linalg.norm(disp_vec, axis=1)
grid_u_warped = grid_u.warp_by_vector(vectors="disp", factor=WARP_SCALE)

# Shared cell-data grid (DG0 — build from mesh, not function space)
c_vm, t_vm, x_vm = vtk_mesh(msh)
grid_vm = pyvista.UnstructuredGrid(c_vm, t_vm, x_vm)
grid_vm.cell_data["von_mises"]     = vm_array
grid_vm.cell_data["max_principal"] = max_principal

cell_centers = grid_vm.cell_centers().points

# Von Mises peak location and clip plane
vm_cell_pt = cell_centers[int(np.argmax(vm_array))]
print(f"Max Von Mises at (x={vm_cell_pt[0]:.2f}, y={vm_cell_pt[1]:.2f}, "
      f"z={vm_cell_pt[2]:.2f}) mm")
grid_vm_clipped = grid_vm.clip(normal="z", origin=vm_cell_pt)
vm_marker       = pyvista.Sphere(radius=0.15, center=vm_cell_pt)

# Max principal stress peak location and clip plane
sp_cell_pt = cell_centers[int(np.argmax(max_principal))]
print(f"Max σ₁ at (x={sp_cell_pt[0]:.2f}, y={sp_cell_pt[1]:.2f}, "
      f"z={sp_cell_pt[2]:.2f}) mm")
grid_sp_clipped = grid_vm.clip(normal="z", origin=sp_cell_pt)
sp_marker       = pyvista.Sphere(radius=0.15, center=sp_cell_pt)

sp_min_val = float(max_principal.min())
sp_max_val = float(max_principal.max())

N_BUCKETS       = 10
sp_bucket_width = (sp_max_val - sp_min_val) / N_BUCKETS if sp_max_val != sp_min_val else 1.0
sp_bucket_idx   = np.clip(
    ((max_principal - sp_min_val) / sp_bucket_width).astype(int), 0, N_BUCKETS - 1
)
sp_bucket_cells = [np.where(sp_bucket_idx == i)[0] for i in range(N_BUCKETS)]

vm_min_val      = float(vm_array.min())
vm_max_val      = float(vm_array.max())
vm_bucket_width = (vm_max_val - vm_min_val) / N_BUCKETS if vm_max_val != vm_min_val else 1.0
vm_bucket_idx   = np.clip(
    ((vm_array - vm_min_val) / vm_bucket_width).astype(int), 0, N_BUCKETS - 1
)
vm_bucket_cells = [np.where(vm_bucket_idx == i)[0] for i in range(N_BUCKETS)]

plotter = pyvista.Plotter(shape=(1, 3))

# ---- Subplot 0: Displacement ----
plotter.subplot(0, 0)
plotter.add_text(f"Displacement magnitude [mm]  (×{WARP_SCALE})", font_size=10)
plotter.add_mesh(grid_u_warped, scalars="disp_mag", cmap="viridis", show_edges=False)
plotter.add_scalar_bar("‖u‖ [mm]")

# ---- Subplot 1: Von Mises ----
plotter.subplot(0, 1)
plotter.add_text("Von Mises — clipped at peak z  (ductile metals)", font_size=10)
plotter.add_mesh(grid_vm_clipped, scalars="von_mises", cmap="hot", show_edges=False)
plotter.add_mesh(vm_marker, color="cyan")
plotter.add_scalar_bar("σ_VM [MPa]", fmt="%.2f")

# ---- Subplot 2: Max principal stress ----
plotter.subplot(0, 2)
plotter.add_text("Max principal stress σ₁ — clipped at peak z  (brittle ceramics)", font_size=10)
plotter.add_mesh(grid_sp_clipped, scalars="max_principal", cmap="hot", show_edges=False)
plotter.add_mesh(sp_marker, color="cyan")
plotter.add_scalar_bar("σ₁ [MPa]", fmt="%.2f")
legend_actor = plotter.add_legend(fracture_legend, bcolor=None, face="rectangle",
                                  size=(0.45, 0.18), loc="upper right")
legend_actor.GetEntryTextProperty().SetFontSize(10)
legend_actor.SetPosition(0.50, 0.80)

plotter.show()

# ============================================================
# SECOND WINDOW: Equipotential stress views
#   Left  — Max principal stress σ₁ bucket view  (brittle fracture criterion)
#   Right — Von Mises stress bucket view          (ductile yield criterion)
# Use ← / → to step through stress buckets.
# ============================================================

plotter2 = pyvista.Plotter(shape=(1, 2))

equi_bucket   = [0]
sp_equi_band2 = [None]
vm_equi_band2 = [None]

# ---- plotter2 Subplot 0: Principal stress equipotential ----
plotter2.subplot(0, 0)
plotter2.add_mesh(
    grid_vm, scalars="max_principal", cmap="hot", opacity=0.15,
    clim=[sp_min_val, sp_max_val], show_edges=False,
)
plotter2.add_scalar_bar("σ₁ [MPa]")

# ---- plotter2 Subplot 1: Von Mises equipotential ----
plotter2.subplot(0, 1)
plotter2.add_mesh(
    grid_vm, scalars="von_mises", cmap="hot", opacity=0.15,
    clim=[vm_min_val, vm_max_val], show_edges=False,
)
plotter2.add_scalar_bar("σ_VM [MPa]")


def _equi_title_sp(bucket):
    lo = sp_min_val + bucket * sp_bucket_width
    hi = lo + sp_bucket_width
    return (f"σ₁ bucket view  (← / → to step)\n"
            f"Bucket {bucket + 1}/{N_BUCKETS}:  {lo:.1f} – {hi:.1f} MPa")


def _equi_title_vm(bucket):
    lo = vm_min_val + bucket * vm_bucket_width
    hi = lo + vm_bucket_width
    return (f"σ_VM bucket view  (← / → to step)\n"
            f"Bucket {bucket + 1}/{N_BUCKETS}:  {lo:.1f} – {hi:.1f} MPa")


def _show_equi_bucket(bucket):
    if sp_equi_band2[0] is not None:
        plotter2.remove_actor(sp_equi_band2[0])
        sp_equi_band2[0] = None
    sp_cells = sp_bucket_cells[bucket]
    if len(sp_cells) > 0:
        plotter2.subplot(0, 0)
        sp_equi_band2[0] = plotter2.add_mesh(
            grid_vm.extract_cells(sp_cells),
            scalars="max_principal", cmap="hot", opacity=1.0,
            clim=[sp_min_val, sp_max_val], show_edges=False, show_scalar_bar=False,
        )
    plotter2.subplot(0, 0)
    plotter2.add_text(_equi_title_sp(bucket), font_size=9, name="sp_equi_title")

    if vm_equi_band2[0] is not None:
        plotter2.remove_actor(vm_equi_band2[0])
        vm_equi_band2[0] = None
    vm_cells = vm_bucket_cells[bucket]
    if len(vm_cells) > 0:
        plotter2.subplot(0, 1)
        vm_equi_band2[0] = plotter2.add_mesh(
            grid_vm.extract_cells(vm_cells),
            scalars="von_mises", cmap="hot", opacity=1.0,
            clim=[vm_min_val, vm_max_val], show_edges=False, show_scalar_bar=False,
        )
    plotter2.subplot(0, 1)
    plotter2.add_text(_equi_title_vm(bucket), font_size=9, name="vm_equi_title")

    plotter2.render()


def _equi_prev():
    equi_bucket[0] = max(0, equi_bucket[0] - 1)
    _show_equi_bucket(equi_bucket[0])


def _equi_next():
    equi_bucket[0] = min(N_BUCKETS - 1, equi_bucket[0] + 1)
    _show_equi_bucket(equi_bucket[0])


_show_equi_bucket(0)

plotter2.add_key_event('Left',  _equi_prev)
plotter2.add_key_event('Right', _equi_next)

plotter2.show()
