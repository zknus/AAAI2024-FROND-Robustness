from .function_laplacian_diffusion import LaplacianODEFunc
from .function_transformer_attention import ODEFuncTransformerAtt
from .function_beltrami_trans import ODEFuncBeltramiAtt
from .function_transformer_grand import ODEFuncTransformerAtt_GRAND
from .block_constant_fractional import ConstantODEblock_FRAC
class BlockNotDefined(Exception):
    pass


class FunctionNotDefined(Exception):
    pass


def set_block(opt):
    ode_str = opt['block']
    if ode_str == 'constantfrac':
        block = ConstantODEblock_FRAC

    else:
        raise BlockNotDefined
    return block


def set_function(opt):
    ode_str = opt['function']
    if ode_str == 'laplacian':
        f = LaplacianODEFunc
    elif ode_str == 'transformer':
        f = ODEFuncTransformerAtt
    elif ode_str == 'belgrand':
        f = ODEFuncBeltramiAtt
    elif ode_str == 'transgrand':
        f = ODEFuncTransformerAtt_GRAND
    else:
        raise FunctionNotDefined
    return f
