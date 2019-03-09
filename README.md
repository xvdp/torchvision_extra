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
## Installation
Similar to benchmark_maskrcnn - minor differences noted here. Testing WIP.
```
# in your conda environment
# ...
conda install ninja yacs cython matplotlib 
conda install cudatoolkit=9 # or =10 ensure cuda tooklit matches your distribution 

# install pytorch and torchvision 
# maskrcnn_benchmark calls for own compilation of vision and for pytorch nightly
# this works with either one
conda install pytorch torchvision -c pytorch


git clone https://github.com/xvdp/nms_pytorch
cd nms_pytorch
python setup.py build develop
```
## Use instructions wip
