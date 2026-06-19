from petsc4py.PETSc import ScalarType  # type: ignore

import numpy as np
import ufl
import pyvista

from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
from dolfinx.plot import vtk_mesh

from piezo_mesh_models import create_piezo_mesh, LN_DZ, LN_LENGTH, E_DZ, ELECTRODE_TAG, LN_TAG

mesh_data = create_piezo_mesh()
msh = mesh_data.mesh

Z_BOTTOM = 0.0
Z_TOP = 16 * (LN_DZ + E_DZ) + LN_DZ  # 5.26 mm

# Center of the top LN layer
SOURCE_X = LN_LENGTH / 2       # 5.0 mm
SOURCE_Y = LN_LENGTH / 2       # 5.0 mm
SOURCE_Z = Z_TOP - LN_DZ / 2   # midpoint in z of the top layer

V = fem.functionspace(msh, ("Lagrange", 1))

tdim = msh.topology.dim
fdim = tdim - 1

# Fix T=0 on the bottom face; all other surfaces are insulated (natural Neumann)
bottom_facets = mesh.locate_entities_boundary(
    msh, dim=fdim, marker=lambda x: np.isclose(x[2], Z_BOTTOM)
)
bottom_dofs = fem.locate_dofs_topological(V=V, entity_dim=fdim, entities=bottom_facets)
bcs = [fem.dirichletbc(value=ScalarType(0), dofs=bottom_dofs, V=V)]

# Point source approximated as a narrow Gaussian (sigma << mesh size)
x = ufl.SpatialCoordinate(msh)
sigma = 0.3  # mm — keep larger than the typical element size
r2 = (x[0] - SOURCE_X)**2 + (x[1] - SOURCE_Y)**2 + (x[2] - SOURCE_Z)**2
f = ufl.exp(-r2 / (2 * sigma**2))

# Piecewise thermal conductivity via a DG0 function (one value per cell)
# LiNbO3: ~4.6 W/(m·K),  Aluminum: ~205 W/(m·K)
K = fem.functionspace(msh, ("DG", 0))
k = fem.Function(K)
k.x.array[mesh_data.cell_tags.find(LN_TAG)]       = 4.6
k.x.array[mesh_data.cell_tags.find(ELECTRODE_TAG)] = 205.0

T = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

a = k * ufl.dot(ufl.grad(T), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx

problem = LinearProblem(
    a, L, bcs=bcs,
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    petsc_options_prefix="heat_",
)
Th = problem.solve()
Th.name = "T"

print(f"Max T: {Th.x.array.real.max():.4f},  Min T: {Th.x.array.real.min():.4f}")

cells, types, coords = vtk_mesh(V)
grid = pyvista.UnstructuredGrid(cells, types, coords)
grid.point_data["T"] = Th.x.array.real

plotter = pyvista.Plotter()
plotter.add_mesh(grid, scalars="T", cmap="hot", show_edges=False)
plotter.add_scalar_bar("T", title="Temperature (a.u.)")
plotter.show()
