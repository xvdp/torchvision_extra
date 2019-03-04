# nms_pytorch
standalone version of nms included with https://github.com/facebookresearch/maskrcnn-benchmark

I needed non maximum suppression outside the scope of maskrcnn so I extracted the nms portion of the code and exposed the functions:

```
.nms()
.prepare_boxlist()
.boxlist_nms()
.remove_small_boxes()
.boxlist_iou()
```
