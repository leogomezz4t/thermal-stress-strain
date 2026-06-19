from mpi4py import MPI

import gmsh  # type: ignore

from dolfinx.io import gmsh as gmshio
import dolfinx.plot as plot
import pyvista

# ALL IN mm
LN_LENGTH = 10
LN_DZ = 0.3

E_DX = 2
E_DY = LN_LENGTH
E_DZ = 0.012

ELECTRODE_TAG = 1
LN_TAG = 2
MACOR_TAG = 3
EPOXY_TAG = 4

tag_colors = {ELECTRODE_TAG: "gray", LN_TAG: "yellow", MACOR_TAG: "blue", EPOXY_TAG: "green"}

def _add_electrode_layer(model: gmsh.model, e_tags, epoxy_tags, x, y, z):
    """Two electrode strips + epoxy filling the gaps, all at height z with thickness E_DZ."""
    left  = model.occ.add_box(x - (E_DX/2) + (1/3)*LN_LENGTH, y, z, E_DX, E_DY, E_DZ)
    right = model.occ.add_box(x - (E_DX/2) + (2/3)*LN_LENGTH, y, z, E_DX, E_DY, E_DZ)
    fill  = model.occ.add_box(x, y,                                 z, LN_LENGTH, LN_LENGTH, E_DZ)
    e_tags.extend([left, right])
    epoxy_tags.append(fill)
    return [left, right, fill]

def create_piezo_layer(model: gmsh.model, e_tags, ln_tags, epoxy_tags, x, y, z):
    ln_box     = model.occ.add_box(x, y, z, LN_LENGTH, LN_LENGTH, LN_DZ)
    ln_tags.append(ln_box)
    elec_tags  = _add_electrode_layer(model, e_tags, epoxy_tags, x, y, z + LN_DZ)
    return [ln_box] + elec_tags

def create_piezo_model(model: gmsh.model, name: str):
    model.add(name)
    model.setCurrent(name)

    all_tags = []
    electrode_tags = []
    ln_tags = []
    macor_tags = []
    epoxy_tags = []

    x, y, z = 0, 0, 0

    # Bottom macor cap
    bottom_macor = model.occ.add_box(x, y, z, LN_LENGTH, LN_LENGTH, LN_DZ)
    macor_tags.append(bottom_macor)
    all_tags.append(bottom_macor)
    z += LN_DZ

    # Bottom electrode pair + epoxy fill
    all_tags.extend(_add_electrode_layer(model, electrode_tags, epoxy_tags, x, y, z))
    z += E_DZ

    # 16 LN layers with top electrodes + epoxy fill
    for i in range(16):
        layer_tags = create_piezo_layer(model, electrode_tags, ln_tags, epoxy_tags, x, y, z)
        all_tags.extend(layer_tags)
        z += LN_DZ + E_DZ

    # Top electrode pair + epoxy fill
    all_tags.extend(_add_electrode_layer(model, electrode_tags, epoxy_tags, x, y, z))
    z += E_DZ

    # Top macor cap
    top_macor = model.occ.add_box(x, y, z, LN_LENGTH, LN_LENGTH, LN_DZ)
    macor_tags.append(top_macor)
    all_tags.append(top_macor)

    # Fragment all volumes so shared faces become conforming (shared nodes).
    # Without this, touching boxes produce a disconnected mesh and a singular system.
    electrode_set = set(electrode_tags)
    macor_set = set(macor_tags)
    epoxy_set = set(epoxy_tags)
    ln_set = set(ln_tags)
    all_dimtags = [(3, t) for t in all_tags]
    _, out_map = model.occ.fragment(all_dimtags, [])

    # Overlapping source volumes (epoxy fill vs electrodes) produce shared child volumes
    # in out_map. Process in priority order so each child is claimed exactly once.
    new_electrode_tags, new_ln_tags, new_macor_tags, new_epoxy_tags = [], [], [], []
    seen = set()
    for priority_set, dest in [
        (electrode_set, new_electrode_tags),
        (macor_set,     new_macor_tags),
        (ln_set,        new_ln_tags),
        (epoxy_set,     new_epoxy_tags),
    ]:
        for i, (_, orig_tag) in enumerate(all_dimtags):
            if orig_tag in priority_set:
                for dt in out_map[i]:
                    t = dt[1]
                    if t not in seen:
                        dest.append(t)
                        seen.add(t)

    model.occ.synchronize()
    model.add_physical_group(dim=3, tags=new_electrode_tags, tag=ELECTRODE_TAG)
    model.add_physical_group(dim=3, tags=new_ln_tags, tag=LN_TAG)
    model.add_physical_group(dim=3, tags=new_macor_tags, tag=MACOR_TAG)
    model.add_physical_group(dim=3, tags=new_epoxy_tags, tag=EPOXY_TAG)
    

    # Generate the mesh
    model.mesh.generate(dim=3)
    return model

def create_piezo_mesh():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0) # Turns off debug ouput to console

    # Create model
    model = gmsh.model()
    model = create_piezo_model(model, "PiezoStack")
    model.setCurrent("PiezoStack")
    mesh_data = create_mesh(MPI.COMM_SELF, model, "PiezoStack")
    gmsh.finalize()
    return mesh_data

def create_mesh(comm: MPI.Comm, model: gmsh.model, name: str):
    mesh_data = gmshio.model_to_mesh(model, comm, rank=0)
    mesh_data.mesh.name = name
    if mesh_data.cell_tags is not None:
        mesh_data.cell_tags.name = f"{name}_cells"
    if mesh_data.facet_tags is not None:
        mesh_data.facet_tags.name = f"{name}_facets"
    if mesh_data.ridge_tags is not None:
        mesh_data.ridge_tags.name = f"{name}_ridges"
    if mesh_data.peak_tags is not None:
        mesh_data.peak_tags.name = f"{name}_peaks"

    return mesh_data

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

if __name__ == "__main__":
    piezo_mesh_data = create_piezo_mesh()
    show_mesh(piezo_mesh_data)