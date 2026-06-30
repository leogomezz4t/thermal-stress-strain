from mesh import Mesh
from materials import Materials
from utils import show_mesh
from stacks.strip_piezo_stack import create_strip_stack, PIEZO_TAG_EVEN, PIEZO_TAG_ODD, CAP_TAG, EPOXY_TAG, ELECTRODE_TAG

# ALL IN mm
LN_LENGTH = 10
LN_DZ = 0.3

M_LENGTH = 10
M_DZ = 0.3

E_DX = 2
E_DY = LN_LENGTH
E_DZ = 0.08

materials = {
    PIEZO_TAG_EVEN:Materials.LITHIUM_NIOBATE_4K,
    PIEZO_TAG_ODD: Materials.LITHIUM_NIOBATE_4K,
    ELECTRODE_TAG: Materials.ALUMINUM_4K,
    CAP_TAG:     Materials.MACOR_4K,
    EPOXY_TAG:     Materials.EPOXY_353ND_4K,
}

class LN_Al_Strip(Mesh):
    def __init__(self):
        msh_data = create_strip_stack(
            LN_LENGTH,
            LN_DZ,
            E_DX,
            E_DY,
            E_DZ,
            M_LENGTH,
            M_DZ,
            n_layers=16,
            n_refinements=1
        )
        super().__init__(
            msh_data,
            0.0,
            16 * (LN_DZ + E_DZ) + 2 * (M_DZ + E_DZ),   # z-coordinate of the top face [mm]
            materials
        )
