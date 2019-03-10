# xvdp modification to facebook file- remove JIT load C as module

import nms_pytorch._C as _C
nms = _C.nms
# nms.__doc__ = """
# This function performs Non-maximum suppresion"""
