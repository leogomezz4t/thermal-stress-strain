from dolfinx import plot
import pyvista
from stacks.full_piezo_stack import ELECTRODE_TAG, PIEZO_TAG, CAP_TAG, EPOXY_TAG

tag_colors = {ELECTRODE_TAG: "gray", PIEZO_TAG: "orange", CAP_TAG: "white", EPOXY_TAG: "yellow"}

def show_mesh(mesh_data):        
    cells, types, x = plot.vtk_mesh(mesh_data.mesh)
    grid = pyvista.UnstructuredGrid(cells, types, x)

    # Plot
    plotter = pyvista.Plotter()
    for tag_id, color in tag_colors.items():
        # Extract only cells with this tag
        mask = mesh_data.cell_tags.values == tag_id
        subgrid = grid.extract_cells(mask)
        plotter.add_mesh(subgrid, color=color, show_edges=True, label=f"Region {tag_id}")
    plotter.show()