# thermal_stress_v2.py
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
import matplotlib.cm as mcm
import matplotlib.colors as mcolors

from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.plot import vtk_mesh

from stacks.LN_Al_strip import LN_Al_Strip
from stacks.LN_SS_strip import LN_SS_Strip
from stacks.LN_Al_full import LN_Al_Full
from stacks.full_piezo_stack import PIEZO_TAG, CAP_TAG, EPOXY_TAG, ELECTRODE_TAG
from mesh import Mesh
from materials import Materials

# ============================================================
# THERMAL LOADING
# ============================================================
T_REF  = 293.0   # Assembly temperature — the stress-free reference state [K]
T_CRYO = 4.0     # Cryogenic operating temperature [K]
# ΔT = T_CRYO - T_REF is applied uniformly; negative means contraction

# ============================================================
# VISUALISATION
# ============================================================
WARP_SCALE = 3000   # Displacement magnification factor for the deformed mesh plot

# ============================================================
# MESH
# ============================================================

piezo_mesh = LN_Al_Full()

MATERIAL_NAMES = {
    PIEZO_TAG:     "LiNbO3",
    ELECTRODE_TAG: "Aluminum",
    CAP_TAG:       "Macor",
    EPOXY_TAG:     "Epoxy 353ND",
}

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

msh  = piezo_mesh.msh
tdim = piezo_mesh.tdim

# Vector Lagrange space: 3 displacement DOFs per mesh node
V_u = fem.functionspace(msh, ("Lagrange", 1, (tdim,)))

# Piecewise Lamé parameters and CTE via DG0 spaces
DG0 = fem.functionspace(msh, ("DG", 0))
lam_fn   = fem.Function(DG0)
mu_fn    = fem.Function(DG0)
alpha_fn = fem.Function(DG0)

# λ = E ν / ((1+ν)(1-2ν))
def lame_lambda(E, nu): return E * nu / ((1 + nu) * (1 - 2*nu))
def lame_mu(E, nu):     return E / (2 * (1 + nu))

for tag, mat in piezo_mesh.materials.items():
    cells = piezo_mesh.cell_tags.find(tag)
    lam_fn.x.array[cells]   = lame_lambda(mat.E, mat.nu)
    mu_fn.x.array[cells]    = lame_mu(mat.E, mat.nu)
    alpha_fn.x.array[cells] = mat.alpha

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
fdim = piezo_mesh.fdim
bottom_dofs_u = fem.locate_dofs_topological(V=V_u, entity_dim=fdim, entities=piezo_mesh.bottom_facets)
top_dofs_u    = fem.locate_dofs_topological(V=V_u, entity_dim=fdim, entities=piezo_mesh.top_facets)
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
# POST-PROCESSING: Maximum principal stress (σ₁)
# Von Mises is the correct failure criterion for ductile metals
# (aluminum), but LiNbO₃ and Macor are brittle ceramics — they
# fracture in tension when σ₁ exceeds the tensile fracture strength.
# Hydrostatic stress contributes zero to Von Mises but fully to σ₁,
# so the two criteria can disagree significantly.
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

# Fracture legend normalised to σ₁ scale — the correct comparison for brittle materials
_hot = mcm.get_cmap("hot")
fracture_legend = [
    [f"{MATERIAL_NAMES[tag]}  σ_f = {mat.sigma_f} MPa",
     "black"]
    for tag, mat in piezo_mesh.materials.items()
    if tag in MATERIAL_NAMES
]

# ============================================================
# VISUALISATION: three panels
#   Left   — displacement magnitude on warped mesh
#   Centre — Von Mises stress with clip plane through the peak
#   Right  — thresholded high-stress region only
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
grid_vm.cell_data["von_mises"]    = vm_array
grid_vm.cell_data["max_principal"] = max_principal

cell_centers = grid_vm.cell_centers().points

# Von Mises peak location and clip plane
vm_cell_pt  = cell_centers[int(np.argmax(vm_array))]
print(f"Max Von Mises at (x={vm_cell_pt[0]:.2f}, y={vm_cell_pt[1]:.2f}, "
      f"z={vm_cell_pt[2]:.2f}) mm")
grid_vm_clipped = grid_vm.clip(normal="z", origin=vm_cell_pt)
vm_marker       = pyvista.Sphere(radius=0.15, center=vm_cell_pt)

# Max principal stress peak location and clip plane
sp_cell_pt  = cell_centers[int(np.argmax(max_principal))]
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
