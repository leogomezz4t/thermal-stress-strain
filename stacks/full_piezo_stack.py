from dataclasses import dataclass
from mpi4py import MPI

import gmsh  # type: ignore
from dolfinx.io import gmsh as gmshio

# Cell tags — fixed, not user-configurable
ELECTRODE_TAG = 1
PIEZO_TAG     = 2
CAP_TAG       = 3
EPOXY_TAG     = 4


@dataclass
class _Dims:
    piezo_length: float  # side length of square piezoelectric slab (shared with electrodes) [mm]
    piezo_dz:     float  # piezoelectric layer thickness [mm]
    e_dz:         float  # electrode layer thickness [mm]
    epoxy_dz:     float  # epoxy bonding layer thickness [mm]
    cap_length:   float  # side length of square cap slab [mm]
    cap_dz:       float  # cap thickness [mm]
    n_layers:     int    # number of piezoelectric + electrode unit cells


def _add_piezo_layer(model, piezo_tags, x, y, z, d: _Dims):
    box = model.occ.add_box(x, y, z, d.piezo_length, d.piezo_length, d.piezo_dz)
    piezo_tags.append(box)
    return box


def _add_electrode_layer(model, e_tags, x, y, z, d: _Dims):
    box = model.occ.add_box(x, y, z, d.piezo_length, d.piezo_length, d.e_dz)
    e_tags.append(box)
    return box


def _add_epoxy_layer(model, epoxy_tags, x, y, z, d: _Dims):
    box = model.occ.add_box(x, y, z, d.piezo_length, d.piezo_length, d.epoxy_dz)
    epoxy_tags.append(box)
    return box


def _build_gmsh_model(name: str, d: _Dims) -> gmsh.model:
    model = gmsh.model()
    model.add(name)
    model.setCurrent(name)

    all_tags       = []
    electrode_tags = []
    piezo_tags     = []
    cap_tags       = []
    epoxy_tags     = []

    x, y, z = 0.0, 0.0, 0.0

    # Bottom cap
    bottom_cap = model.occ.add_box(x, y, z, d.cap_length, d.cap_length, d.cap_dz)
    cap_tags.append(bottom_cap)
    all_tags.append(bottom_cap)
    z += d.cap_dz

    # Epoxy
    all_tags.append(_add_epoxy_layer(model, epoxy_tags, x, y, z, d))
    z += d.epoxy_dz

    # Bottom electrode
    all_tags.append(_add_electrode_layer(model, electrode_tags, x, y, z, d))
    z += d.e_dz

    # n_layers unit cells: epoxy + piezoelectric slab + epoxy + electrode
    for _ in range(d.n_layers):
        all_tags.append(_add_epoxy_layer(model, epoxy_tags, x, y, z, d))
        z += d.epoxy_dz

        all_tags.append(_add_piezo_layer(model, piezo_tags, x, y, z, d))
        z += d.piezo_dz

        all_tags.append(_add_epoxy_layer(model, epoxy_tags, x, y, z, d))
        z += d.epoxy_dz

        all_tags.append(_add_electrode_layer(model, electrode_tags, x, y, z, d))
        z += d.e_dz

    # Epoxy
    all_tags.append(_add_epoxy_layer(model, epoxy_tags, x, y, z, d))
    z += d.epoxy_dz

    # Top electrode
    # all_tags.append(_add_electrode_layer(model, electrode_tags, x, y, z, d))
    # z += d.e_dz

    # Epoxy
    all_tags.append(_add_epoxy_layer(model, epoxy_tags, x, y, z, d))
    z += d.epoxy_dz

    # Top cap
    top_cap = model.occ.add_box(x, y, z, d.cap_length, d.cap_length, d.cap_dz)
    cap_tags.append(top_cap)
    all_tags.append(top_cap)

    # Fragment all volumes so shared faces become conforming (shared nodes).
    # Without this, touching boxes produce a disconnected mesh and a singular system.
    electrode_set = set(electrode_tags)
    cap_set       = set(cap_tags)
    piezo_set     = set(piezo_tags)
    epoxy_set     = set(epoxy_tags)

    all_dimtags = [(3, t) for t in all_tags]
    _, out_map  = model.occ.fragment(all_dimtags, [])

    new_electrode_tags, new_piezo_tags, new_cap_tags, new_epoxy_tags = [], [], [], []
    seen = set()
    for priority_set, dest in [
        (electrode_set, new_electrode_tags),
        (cap_set,       new_cap_tags),
        (piezo_set,     new_piezo_tags),
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
    model.add_physical_group(dim=3, tags=new_piezo_tags,     tag=PIEZO_TAG)
    model.add_physical_group(dim=3, tags=new_cap_tags,       tag=CAP_TAG)
    model.add_physical_group(dim=3, tags=new_epoxy_tags,     tag=EPOXY_TAG)

    model.mesh.generate(dim=3)
    return model


def _model_to_mesh(model: gmsh.model, name: str, comm: MPI.Comm):
    mesh_data = gmshio.model_to_mesh(model, comm, rank=0)
    mesh_data.mesh.name = name
    if mesh_data.cell_tags  is not None: mesh_data.cell_tags.name  = f"{name}_cells"
    if mesh_data.facet_tags is not None: mesh_data.facet_tags.name = f"{name}_facets"
    if mesh_data.ridge_tags is not None: mesh_data.ridge_tags.name = f"{name}_ridges"
    if mesh_data.peak_tags  is not None: mesh_data.peak_tags.name  = f"{name}_peaks"
    return mesh_data


def create_full_stack(
    piezo_length: float,
    piezo_dz:     float,
    e_dz:         float,
    cap_length:   float,
    cap_dz:       float,
    epoxy_dz:     float = 0.001,
    n_layers:     int   = 16,
    name:         str   = "FullPiezoStack",
):
    """Build and mesh a full-electrode piezo stack with epoxy bonding layers.

    Electrodes span the full piezo_length × piezo_length square. A thin epoxy
    layer is placed between every adjacent pair of layers. All dimensions in mm.
    Returns a dolfinx MeshData object whose cell tags use ELECTRODE_TAG,
    PIEZO_TAG, CAP_TAG, EPOXY_TAG.

    Parameters
    ----------
    piezo_length  Side length of the square piezoelectric slab (also the electrode).
    piezo_dz      Thickness of each piezoelectric layer.
    e_dz          Thickness of each electrode layer.
    cap_length    Side length of the square cap slab.
    cap_dz        Thickness of each cap.
    epoxy_dz      Thickness of each epoxy bonding layer.
    n_layers      Number of piezoelectric + electrode unit cells.
    name          Name given to the gmsh model and dolfinx mesh.
    """
    dims = _Dims(
        piezo_length=piezo_length,
        piezo_dz=piezo_dz,
        e_dz=e_dz,
        epoxy_dz=epoxy_dz,
        cap_length=cap_length,
        cap_dz=cap_dz,
        n_layers=n_layers,
    )

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        model     = _build_gmsh_model(name, dims)
        mesh_data = _model_to_mesh(model, name, MPI.COMM_SELF)
    finally:
        gmsh.finalize()

    return mesh_data
