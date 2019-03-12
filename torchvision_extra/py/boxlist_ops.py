#  xvdp Modifiedfrom maskrcnn_benchmark (Copyright (c) Facebook)
# from boxlist_ops.py 
# and inference.py 
#   prepare_boxlist() ( from PostProcessor class )
import torch

from .bounding_box import BoxList
from .nms import nms


    #     boxlist = self.prepare_boxlist(boxes_per_img, prob, image_shape)
    #     boxlist = boxlist.clip_to_image(remove_empty=False)
    #     boxlist = self.filter_results(boxlist, num_classes)
    #     results.append(boxlist)
    # return results

def prepare_boxlist(boxes, scores, image_shape, mode="xyxy"):
    """
    modified xvdp added argument 'mode'

    Arguments:
        `boxes`       shape (#detections, 4 * #classes)
            each row represents a list of predicted bounding boxes
            for each of the object classes in the dataset (including the background class).
            The detections in each row originate from the same object proposal.
        `scores`      shape (#detection, #classes), or (#detection) if only one class
            each row represents a list of object detection confidence scores
            for each of the object classes in the dataset (including the background class). 
            `scores[i, j]` corresponds to the box at `boxes[i, j * 4:(j + 1) * 4]`.
        `image_shape` tuple int
        `mode`        str, "xyxy"

    Returns:
        BoxList from `boxes` and adds probability scores information as an extra field
    """
    boxes = boxes.reshape(-1, 4)
    scores = scores.reshape(-1)
    boxlist = BoxList(boxes, image_shape, mode=mode)
    boxlist.add_field("scores", scores)
    return boxlist

    #TODO expose
    # def filter_results(self, boxlist, num_classes):
    #     """Returns bounding-box detection results by thresholding on scores and
    #     applying non-maximum suppression (NMS).
    #     """
    #     # unwrap the boxlist to avoid additional overhead.
    #     # if we had multi-class NMS, we could perform this directly on the boxlist
    #     boxes = boxlist.bbox.reshape(-1, num_classes * 4)
    #     scores = boxlist.get_field("scores").reshape(-1, num_classes)

    #     device = scores.device
    #     result = []
    #     # Apply threshold on detection probabilities and apply NMS
    #     # Skip j = 0, because it's the background class
    #     inds_all = scores > self.score_thresh
    #     for j in range(1, num_classes):
    #         inds = inds_all[:, j].nonzero().squeeze(1)
    #         scores_j = scores[inds, j]
    #         boxes_j = boxes[inds, j * 4 : (j + 1) * 4]
    #         boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
    #         boxlist_for_class.add_field("scores", scores_j)
    #         boxlist_for_class = boxlist_nms(
    #             boxlist_for_class, self.nms
    #         )
    #         num_labels = len(boxlist_for_class)
    #         boxlist_for_class.add_field(
    #             "labels", torch.full((num_labels,), j, dtype=torch.int64, device=device)
    #         )
    #         result.append(boxlist_for_class)

        # result = cat_boxlist(result)
        # number_of_detections = len(result)

        # # Limit to max_per_image detections **over all classes**
        # if number_of_detections > self.detections_per_img > 0:
        #     cls_scores = result.get_field("scores")
        #     image_thresh, _ = torch.kthvalue(
        #         cls_scores.cpu(), number_of_detections - self.detections_per_img + 1
        #     )
        #     keep = cls_scores >= image_thresh.item()
        #     keep = torch.nonzero(keep).squeeze(1)
        #     result = result[keep]
        # return result


def boxlist_nms(boxlist, nms_thresh, max_proposals=-1, score_field="scores"):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Arguments:
        boxlist(BoxList)
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        score_field (str)
    """
    if nms_thresh <= 0:
        return boxlist
    mode = boxlist.mode
    boxlist = boxlist.convert("xyxy")
    boxes = boxlist.bbox
    score = boxlist.get_field(score_field)
    keep = nms(boxes, score, nms_thresh)
    if max_proposals > 0:
        keep = keep[: max_proposals]
    boxlist = boxlist[keep]
    return boxlist.convert(mode)


def remove_small_boxes(boxlist, min_size):
    """
    Only keep boxes with both sides >= min_size

    Arguments:
        boxlist (Boxlist)
        min_size (int)
    """
    # TODO maybe add an API for querying the ws / hs
    xywh_boxes = boxlist.convert("xywh").bbox
    _, _, ws, hs = xywh_boxes.unbind(dim=1)
    keep = (
        (ws >= min_size) & (hs >= min_size)
    ).nonzero().squeeze(1)
    return boxlist[keep]


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def boxlist_iou(boxlist1, boxlist2):
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Arguments:
      box1: (BoxList) bounding boxes, sized [N,4].
      box2: (BoxList) bounding boxes, sized [M,4].

    Returns:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    if boxlist1.size != boxlist2.size:
        raise RuntimeError(
                "boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))

    N = len(boxlist1)
    M = len(boxlist2)

    area1 = boxlist1.area()
    area2 = boxlist2.area()

    box1, box2 = boxlist1.bbox, boxlist2.bbox

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    TO_REMOVE = 1

    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


# TODO redundant, remove
def _cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def cat_boxlist(bboxes):
    """
    Concatenates a list of BoxList (having the same image size) into a
    single BoxList

    Arguments:
        bboxes (list[BoxList])
    """
    assert isinstance(bboxes, (list, tuple))
    assert all(isinstance(bbox, BoxList) for bbox in bboxes)

    size = bboxes[0].size
    assert all(bbox.size == size for bbox in bboxes)

    mode = bboxes[0].mode
    assert all(bbox.mode == mode for bbox in bboxes)

    fields = set(bboxes[0].fields())
    assert all(set(bbox.fields()) == fields for bbox in bboxes)

    cat_boxes = BoxList(_cat([bbox.bbox for bbox in bboxes], dim=0), size, mode)

    for field in fields:
        data = _cat([bbox.get_field(field) for bbox in bboxes], dim=0)
        cat_boxes.add_field(field, data)

    return cat_boxes
