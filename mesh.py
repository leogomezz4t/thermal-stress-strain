from dataclasses import dataclass
from typing import Dict

import numpy as np
from dolfinx import mesh as dmesh
from materials import MaterialProperties



class Mesh:
    """Wraps a dolfinx mesh with its boundary geometry and per-material properties.

    Parameters
    ----------
    mesh_data:  return value of dolfinx's gmshio.model_to_mesh — carries .mesh and .cell_tags
    z_bottom:   z-coordinate of the bottom clamped face [mm]
    z_top:      z-coordinate of the top clamped face [mm]
    materials:  mapping from cell-tag integer → MaterialProperties
    """

    def __init__(
        self,
        mesh_data,
        z_bottom: float,
        z_top: float,
        materials: Dict[int, MaterialProperties],
    ):
        self.mesh_data = mesh_data
        self.msh = mesh_data.mesh
        self.cell_tags = mesh_data.cell_tags
        self.z_bottom = z_bottom
        self.z_top = z_top
        self.materials = materials

    @property
    def tdim(self) -> int:
        return self.msh.topology.dim

    @property
    def fdim(self) -> int:
        return self.tdim - 1

    @property
    def bottom_facets(self):
        return dmesh.locate_entities_boundary(
            self.msh, dim=self.fdim, marker=lambda x: np.isclose(x[2], self.z_bottom)
        )

    @property
    def top_facets(self):
        return dmesh.locate_entities_boundary(
            self.msh, dim=self.fdim, marker=lambda x: np.isclose(x[2], self.z_top)
        )
