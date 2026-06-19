# thermal_stress_v1.py
#
# Thermoelastic model of the piezo stack under uniform cooling to cryogenic temperature.
# Temperature is uniform (T_CRYO everywhere), so no heat equation is needed —
# ΔT = T_CRYO - T_REF is applied directly as a constant field.
# Stress arises from differential thermal contraction between the four materials:
#   LiNbO3 layers, aluminum electrodes, Macor caps, and 353ND epoxy fill.
#
# LiNbO3 is treated as isotropic (v1 simplification).
# Mesh coordinates are in mm, so stress is in MPa (N/mm²).
#
# Boundary conditions:
#   Mechanical — top and bottom faces clamped (u=0), all other faces traction-free

from petsc4py.PETSc import ScalarType  # type: ignore
import numpy as np
import ufl
import pyvista

from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
from dolfinx.plot import vtk_mesh

from piezo_mesh_models import (
    create_piezo_mesh, LN_DZ, E_DZ, ELECTRODE_TAG, LN_TAG, MACOR_TAG, EPOXY_TAG
)

# ============================================================
# MECHANICAL PROPERTIES
# E in MPa (N/mm²), dimensionless ν, α in 1/K
# ============================================================
LN_E     = 170_000   # LiNbO3 Young's modulus  [MPa]
LN_NU    = 0.25      # LiNbO3 Poisson's ratio
LN_ALPHA = 4e-6      # LiNbO3 coefficient of thermal expansion [1/K]

AL_E     = 70_000    # Aluminum Young's modulus  [MPa]
AL_NU    = 0.33      # Aluminum Poisson's ratio
AL_ALPHA = 23e-6     # Aluminum coefficient of thermal expansion [1/K]

# Note: properties below are room-temperature values; both α and E change
# significantly at cryogenic temperatures and should be updated for accuracy.
MACOR_E     = 66_900   # Macor Young's modulus  [MPa]
MACOR_NU    = 0.29     # Macor Poisson's ratio
MACOR_ALPHA = 9.3e-6   # Macor CTE [1/K]

EPOXY_E     = 3_500    # 353ND epoxy Young's modulus  [MPa]
EPOXY_NU    = 0.35     # 353ND epoxy Poisson's ratio
EPOXY_ALPHA = 54e-6    # 353ND epoxy CTE [1/K]

# ============================================================
# THERMAL LOADING
# ============================================================
T_REF  = 293.0   # Assembly temperature — the stress-free reference state [K]
T_CRYO = 4.0     # Cryogenic operating temperature [K]
# ΔT = T_CRYO - T_REF is applied uniformly; negative means contraction

# ============================================================
# VISUALISATION
# ============================================================
WARP_SCALE = 100   # Displacement magnification factor for the deformed mesh plot

# ============================================================
# MESH
# ============================================================
mesh_data  = create_piezo_mesh()
msh        = mesh_data.mesh
cell_tags  = mesh_data.cell_tags

Z_BOTTOM  = 0.0
# Stack: bottom macor + bottom electrode + 16×(LN + electrode) + top electrode + top macor
Z_TOP     = 18 * (LN_DZ + E_DZ)   # z-coordinate of the top face [mm]

tdim = msh.topology.dim   # 3
fdim = tdim - 1            # 2 (surface facets)

# Locate top and bottom boundary facets for the mechanical clamping BCs
bottom_facets = mesh.locate_entities_boundary(
    msh, dim=fdim, marker=lambda x: np.isclose(x[2], Z_BOTTOM)
)
top_facets = mesh.locate_entities_boundary(
    msh, dim=fdim, marker=lambda x: np.isclose(x[2], Z_TOP)
)

# ============================================================
# THERMAL STRESS
# Solve  -∇·σ = 0
#   σ = C : (ε(u) - ε_th)
#   ε_th = α ΔT I   (isotropic thermal strain)
#   ΔT = T(x) - T_REF
#
#   Weak form:
#   ∫ σ_elastic(u) : ε(v) dx = ∫ (3λ+2μ) α ΔT div(v) dx
#
#   u = 0  on top and bottom faces (clamped)
#   σ·n = 0 on all other faces (traction-free, natural Neumann)
# ============================================================

# Vector Lagrange space: 3 displacement DOFs per mesh node
V_u = fem.functionspace(msh, ("Lagrange", 1, (tdim,)))

# Piecewise Lamé parameters and CTE via DG0 spaces
DG0 = fem.functionspace(msh, ("DG", 0))
lam_fn   = fem.Function(DG0)
mu_fn    = fem.Function(DG0)
alpha_fn = fem.Function(DG0)

ln_cells    = cell_tags.find(LN_TAG)
al_cells    = cell_tags.find(ELECTRODE_TAG)
macor_cells = cell_tags.find(MACOR_TAG)
epoxy_cells = cell_tags.find(EPOXY_TAG)

# λ = E ν / ((1+ν)(1-2ν))
def lame_lambda(E, nu): return E * nu / ((1 + nu) * (1 - 2*nu))
def lame_mu(E, nu):     return E / (2 * (1 + nu))

for cells, E, nu, alpha in [
    (ln_cells,    LN_E,    LN_NU,    LN_ALPHA),
    (al_cells,    AL_E,    AL_NU,    AL_ALPHA),
    (macor_cells, MACOR_E, MACOR_NU, MACOR_ALPHA),
    (epoxy_cells, EPOXY_E, EPOXY_NU, EPOXY_ALPHA),
]:
    lam_fn.x.array[cells]   = lame_lambda(E, nu)
    mu_fn.x.array[cells]    = lame_mu(E, nu)
    alpha_fn.x.array[cells] = alpha

# Uniform temperature change from assembly to cryogenic operating temperature
# Negative value → contraction; differential contraction between materials drives stress
dT = fem.Constant(msh, ScalarType(T_CRYO - T_REF))

# Symmetric strain tensor  ε(u) = ½(∇u + ∇uᵀ)
def eps(u):
    return ufl.sym(ufl.grad(u))

# Elastic stress  σ_el = λ tr(ε) I + 2μ ε
def sigma_elastic(u):
    return lam_fn * ufl.tr(eps(u)) * ufl.Identity(tdim) + 2 * mu_fn * eps(u)

# Full stress including thermal eigenstrain:  σ = C:(ε - α ΔT I)
def sigma_total(u):
    eps_mech = eps(u) - alpha_fn * dT * ufl.Identity(tdim)
    return lam_fn * ufl.tr(eps_mech) * ufl.Identity(tdim) + 2 * mu_fn * eps_mech

u = ufl.TrialFunction(V_u)
v = ufl.TestFunction(V_u)

# Stiffness: ∫ σ_elastic(u) : ε(v) dx
a_u = ufl.inner(sigma_elastic(u), eps(v)) * ufl.dx

# Thermal load: ∫ (3λ+2μ) α ΔT div(v) dx
# This is the contraction of the thermal stress tensor C:ε_th with ε(v)
L_u = (3*lam_fn + 2*mu_fn) * alpha_fn * dT * ufl.div(v) * ufl.dx

# Dirichlet BC: top and bottom faces fully clamped (u=0)
"""
bottom_dofs_u = fem.locate_dofs_topological(V=V_u, entity_dim=fdim, entities=bottom_facets)
top_dofs_u    = fem.locate_dofs_topological(V=V_u, entity_dim=fdim, entities=top_facets)
bcs_u = [
    fem.dirichletbc(value=np.zeros(tdim, dtype=ScalarType), dofs=bottom_dofs_u, V=V_u),
    fem.dirichletbc(value=np.zeros(tdim, dtype=ScalarType), dofs=top_dofs_u,    V=V_u),
]
"""

stress_problem = LinearProblem(
    a_u, L_u, #bcs=bcs_u,
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    petsc_options_prefix="stress_",
)
uh = stress_problem.solve()
uh.name = "u"

disp_mag = np.linalg.norm(uh.x.array.real.reshape(-1, tdim), axis=1)
print(f"Displacement — max magnitude: {disp_mag.max():.4e} mm")

# ============================================================
# POST-PROCESSING: Von Mises stress
# σ_VM = sqrt(3/2 · s:s)   where s = σ - (1/3) tr(σ) I  (deviatoric part)
# High Von Mises stress indicates regions at risk of yielding.
# ============================================================

W = fem.functionspace(msh, ("DG", 0))

sig     = sigma_total(uh)
s       = sig - (1/3) * ufl.tr(sig) * ufl.Identity(tdim)
vm_expr = ufl.sqrt(3/2 * ufl.inner(s, s))

von_mises = fem.Function(W)
von_mises.interpolate(fem.Expression(vm_expr, W.element.interpolation_points))
von_mises.name = "von_mises"

vm_array = von_mises.x.array.real
max_vm   = vm_array.max()
print(f"Von Mises stress — max: {max_vm:.2f} MPa")

# ============================================================
# VISUALISATION: three panels
#   Left   — displacement magnitude on warped mesh
#   Centre — Von Mises stress with clip plane through the peak
#   Right  — thresholded high-stress region only
# ============================================================

# Fraction of max stress below which cells are hidden in the threshold panel
THRESHOLD_FRACTION = 0.85

# Displacement grid (vector, point data) — warp for visibility
c_u, t_u, x_u = vtk_mesh(V_u)
grid_u = pyvista.UnstructuredGrid(c_u, t_u, x_u)
disp_vec = uh.x.array.real.reshape(-1, tdim)
grid_u.point_data["disp"]     = disp_vec
grid_u.point_data["disp_mag"] = np.linalg.norm(disp_vec, axis=1)
grid_u_warped = grid_u.warp_by_vector(vectors="disp", factor=WARP_SCALE)

# Von Mises grid (scalar, cell data on DG0 — build from mesh, not function space)
c_vm, t_vm, x_vm = vtk_mesh(msh)
grid_vm = pyvista.UnstructuredGrid(c_vm, t_vm, x_vm)
grid_vm.cell_data["von_mises"] = vm_array

# Find the centroid of the cell with maximum Von Mises stress
max_cell_idx  = int(np.argmax(vm_array))
max_cell_pt   = grid_vm.cell_centers().points[max_cell_idx]
print(f"Max Von Mises at (x={max_cell_pt[0]:.2f}, y={max_cell_pt[1]:.2f}, "
      f"z={max_cell_pt[2]:.2f}) mm")

# Clip plane: cut the mesh open at the z-coordinate of the peak stress cell,
# revealing the internal cross-section where the maximum occurs
grid_vm_clipped = grid_vm.clip(normal="z", origin=max_cell_pt)

# Threshold: keep only cells above THRESHOLD_FRACTION of the peak stress,
# isolating the high-stress region regardless of where it is in the model
grid_vm_thresh = grid_vm.threshold(value=THRESHOLD_FRACTION * max_vm,
                                   scalars="von_mises")

# Small sphere marking the exact location of the peak
marker = pyvista.Sphere(radius=0.15, center=max_cell_pt)

plotter = pyvista.Plotter(shape=(1, 3))

plotter.subplot(0, 0)
plotter.add_text(f"Displacement magnitude [mm]  (×{WARP_SCALE})", font_size=10)
plotter.add_mesh(grid_u_warped, scalars="disp_mag", cmap="viridis", show_edges=False)
plotter.add_scalar_bar("‖u‖ [mm]")

plotter.subplot(0, 1)
plotter.add_text("Von Mises — clipped at peak z", font_size=10)
plotter.add_mesh(grid_vm_clipped, scalars="von_mises", cmap="hot", show_edges=False)
plotter.add_mesh(marker, color="cyan")   # marks exact peak location
plotter.add_scalar_bar("σ_VM [MPa]")

plotter.subplot(0, 2)
plotter.add_text(f"Von Mises — top {int((1-THRESHOLD_FRACTION)*100)}% stress only",
                 font_size=10)
plotter.add_mesh(grid_vm_thresh, scalars="von_mises", cmap="hot", show_edges=False)
plotter.add_mesh(marker, color="cyan")
plotter.add_scalar_bar("σ_VM [MPa]")

plotter.show()
