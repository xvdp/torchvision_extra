# xvdp modification to facebook file- remove JIT load C as module

import torchvision_extra._C as _C
_nms = _C.nms

from .boxlist_ops import boxlist_nms, prepare_boxlist


def nms(boxes, scores, shape, threshold=0.5, mode="xyxy"):
    """
        wrapper over boxlist nms, moved .nms from _C to ._nms
        Returns boxes, scores tuple after nms
        Arguments:
            boxes:      tensor of shape (N,4)
            scores:     tensor of shape (N,1) or (N)
            shape:      shape of the image
            threshold:  float 0.-1. [0.5]
            mode:       xyxy, xywh, yxyx, yxwh
    """
    boxlist = prepare_boxlist(boxes, scores, shape, mode=mode)
    boxnms = boxlist_nms(boxlist, threshold)

    return boxnms.bbox, boxnms.extra_fields["scores"]
