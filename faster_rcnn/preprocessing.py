"""Data preprocessing operations."""

import numpy as np

from skimage.transform import resize


def resize_by_shorter(image, annotation, shorter_length=600, mode="reflect"):
    """Resize an image such that its shorter dimension is `shorter_length`."""
    height, width, _ = image.shape

    shorter = min(height, width)
    longer = max(height, width)

    ratio = shorter_length / shorter
    longer_length = int(longer * ratio)

    if height > width:
        resized = resize(image, (longer_length, shorter_length), mode=mode)
    else:
        resized = resize(image, (shorter_length, longer_length), mode=mode)

    annotation = annotation.copy()
    for a in annotation:
        a["object"]["bbox"] = [int(n * ratio) for n in a["object"]["bbox"]]
        a["subject"]["bbox"] = [int(n * ratio) for n in a["subject"]["bbox"]]

    return resized, annotation


def generate_anchors(image,
                     stride=16,
                     sides=[128, 256, 512],
                     ratios=[0.5, 1, 2]):
    """Generate anchors to serve as regression targets."""
    imheight, imwidth, _ = image.shape

    # get anchor center coordinates
    y = np.arange(0, imheight, stride)
    x = np.arange(0, imwidth, stride)

    yv, xv = np.meshgrid(y, x)

    y_ctr = np.broadcast_to(yv, (len(sides) * len(ratios), *yv.shape)).T
    x_ctr = np.broadcast_to(xv, (len(sides) * len(ratios), *xv.shape)).T

    # get anchor heights and widths
    h = []
    w = []
    for side in sides:
        for ratio in ratios:
            h.append(int(side * np.sqrt(ratio)))
            w.append(int(side / np.sqrt(ratio)))

    heights = np.broadcast_to(h, (len(y), len(x), len(h)))
    widths = np.broadcast_to(w, (len(y), len(x), len(w)))

    # stack together so that the last dimension matches RPN output
    anchors = np.stack([y_ctr, x_ctr, heights, widths], axis=-1)
    anchors = anchors.reshape(*y_ctr.shape[:2], len(sides) * len(ratios), 4)
    return anchors


def calculate_area(coords):
    """Given box coordinates (ymin, ymax, xmin,  xmax) calculate box area."""
    return max(0, coords[1] - coords[0]) * max(0, coords[3] - coords[2])


def yxhw_to_yyxx(box):
    """Change box representation from yxhw to (ymin, ymax, xmin, xmax)."""
    y, x, h, w = box
    return y - h // 2, y + h // 2, x - w // 2, x + w // 2


def yyxx_to_yxhw(coords):
    """Change box representation from (ymin, ymax, xmin, xmax) to yxhw."""
    ymin, ymax, xmin, xmax = coords
    return (ymin + ymax) // 2, (xmin + xmax) // 2, ymax - ymin, xmax - xmin


def calculate_iou(box_a, box_b):
    """Calculate IoU between two bboxes.

    coords given as (ymin, ymax, xmin, xmax)
    """
    intersect_coords = [max(box_a[0], box_b[0]),
                        min(box_a[1], box_b[1]),
                        max(box_a[2], box_b[2]),
                        min(box_a[3], box_b[3])]

    intersect_area = calculate_area(intersect_coords)
    anchor_area = calculate_area(box_a)
    bbox_area = calculate_area(box_b)

    union_area = anchor_area + bbox_area - intersect_area

    return intersect_area / union_area


def ground_truth_offset(region_coords, bbox_coords):
    """Give the region-parameterized coordinates of a bounding box."""
    ya, xa, wa, ha = yyxx_to_yxhw(region_coords)
    y, x, w, h = yyxx_to_yxhw(bbox_coords)

    ty = (y - ya) / ha
    tx = (x - xa) / wa
    th = np.log(h / ha)
    tw = np.log(w / wa)

    return ty, tx, th, tw


def sample_minibatch(cls_target,
                     reg_target,
                     batch_size,
                     max_positive):
    """From a set of positive and negative labels, sample a certain amount."""
    pos_batch = np.stack(np.where(cls_target == 1), axis=-1)
    num_positive = len(pos_batch)
    neg_batch = np.stack(np.where(cls_target == 0), axis=-1)
    num_negative = len(neg_batch)

    final_olabel = np.zeros_like(cls_target) - 1
    final_clabel = np.zeros_like(reg_target) - 1

    if num_positive > max_positive:
        indices = np.arange(num_positive)
        pos_batch = pos_batch[np.random.choice(indices, max_positive, False)]

    max_negative = batch_size - min(num_positive, max_positive)
    indices = np.arange(num_negative)
    neg_batch = neg_batch[np.random.choice(indices, max_negative, False)]

    for i in pos_batch:
        final_olabel[tuple(i)] = 1
        final_clabel[tuple(i)] = reg_target[tuple(i)]
    for i in neg_batch:
        final_olabel[tuple(i)] = 0

    loss_mask = np.int64(final_olabel > -1)
    loss_mask = np.expand_dims(loss_mask, -1)

    return final_olabel, final_clabel, loss_mask


def generate_rpn_targets(anchors,
                         annotation,
                         image_shape,
                         pos_threshold=0.7,
                         neg_threshold=0.3,
                         batch_size=256,
                         max_positive=128):
    """Generate minibatch of anchor-parameterized coordinates and labels."""
    # initialize ground truth labels to -1
    cls_target = np.zeros((anchors.shape[0:3])) - 1
    reg_target = np.zeros(anchors.shape) - 1

    # first gather bboxes
    otypes = ["subject", "object"]
    bboxes = [rel[otype]["bbox"] for otype in otypes for rel in annotation]
    # shuffle them so that order of overlapping examples doesn't introduce bias
    np.random.shuffle(bboxes)

    # overlapping examples from later bboxes will overwrite earlier ones
    for bbox in bboxes:
        max_iou = 0
        max_index = None
        max_anchor = None
        # iterate over each 4 tuple of anchor coords
        for index in np.ndindex(anchors.shape[0:3]):
            anchor = yxhw_to_yyxx(anchors[index])

            # filter out anchors with centers outside bbox
            if not (bbox[0] < anchor[0] < bbox[1] or
                    bbox[2] < anchor[1] < bbox[3]):
                continue

            # filter out anchors that cross image boundary
            cross_boundary_conditions = (anchor[0] < 0 or
                                         anchor[2] < 0 or
                                         anchor[1] > image_shape[0] or
                                         anchor[3] > image_shape[1])

            if cross_boundary_conditions:
                continue

            iou = calculate_iou(anchor, bbox)
            # condition 1:  iou > threshold
            if iou > pos_threshold:
                cls_target[index] = 1
                reg_target[index] = ground_truth_offset(anchor, bbox)
            elif 0 < iou < neg_threshold:
                cls_target[index] = 0

            # condition 2: highest iou with bbox
            if iou > max_iou:
                max_iou = iou
                max_index = index
                max_anchor = anchor
        cls_target[max_index] = 1
        reg_target[max_index] = ground_truth_offset(max_anchor, bbox)

    # reduce batch size
    cls_target, reg_target, loss_mask = sample_minibatch(cls_target,
                                                         reg_target,
                                                         batch_size,
                                                         max_positive)

    return cls_target, reg_target, loss_mask


def rpn_preprocessing(image, annotation):
    """Wrap data processing functions for RPN input."""
    image, annotation = resize_by_shorter(image, annotation)

    anchors = generate_anchors(image)

    cls_target, reg_target, loss_mask = generate_rpn_targets(
        anchors, annotation, image.shape)

    return image, annotation, anchors, cls_target, reg_target, loss_mask


def generate_detector_targets(image,
                              annotation,
                              proposals,
                              n_categories=100,
                              pos_threshold=0.5,
                              neg_threshold=0.1):
    """Generate minibatch of region-parameterized coordinates and labels.

    Proposals are normalized by image size, and given as [y1, x1, y2, x2].
    """
    imheight, imwidth, _ = image.shape
    cls_target = np.zeros(proposals.shape[0]) - 1
    reg_target = np.zeros((proposals.shape[0], n_categories, 4))
    # get bboxes and categories
    bboxes = []
    categories = []
    for rel in annotation:
        for otype in ["subject", "object"]:
            bboxes.append(rel[otype]["bbox"])
            categories.append(rel[otype]["category"])
    # match each proposal with the bbox it has the highest IoU with
    for index in range(proposals.shape[0]):
        max_iou = 0
        max_category = None
        max_bbox = None
        proposal = proposals[index]
        ymin = proposal[0] * imheight
        ymax = proposal[2] * imheight
        xmin = proposal[1] * imwidth
        xmax = proposal[3] * imwidth
        region = [ymin, ymax, xmin, xmax]
        for bbox, category in zip(bboxes, categories):
            iou = calculate_iou(region, bbox)
            if iou > max_iou:
                max_iou = iou
                max_category = category
                max_bbox = bbox
        if max_iou > pos_threshold:
            cls_target[index] = max_category + 1
            reg_target[index][max_category] = ground_truth_offset(region,
                                                                  max_bbox)
        elif max_iou > neg_threshold:
            cls_target[index] = 0
    return cls_target, reg_target


def postprocess_rcnn_output(image,
                            proposals,
                            rcnn_cls_output,
                            rcnn_reg_output,
                            n=5):
    """Get top-n category labels and bbox coordinates from rcnn output."""
    imheight, imwidth, _ = image.shape

    categories = np.argmax(rcnn_cls_output, axis=-1) - 1

    # filter out predictions indicating no object (background)
    probabilities = np.max(rcnn_cls_output, axis=-1)[categories > -1]
    bboxes = rcnn_reg_output[categories > -1]
    proposals = proposals[categories > -1]
    categories = categories[categories > -1]

    # get top n objects by confidence
    top_n_indices = np.argsort(probabilities)[:n]
    top_n_proposals = proposals[top_n_indices]
    top_n_categories = categories[top_n_indices]
    top_n_bboxes = bboxes[[top_n_indices, top_n_categories]]

    for i in range(top_n_bboxes.shape[0]):
        proposal = top_n_proposals[i]
        bbox = top_n_bboxes[i]

        # unnormalize proposal, get in yxhw format
        ymin = proposal[0] * imheight
        ymax = proposal[2] * imheight
        xmin = proposal[1] * imwidth
        xmax = proposal[3] * imwidth

        a_y, a_x, a_h, a_w = yyxx_to_yxhw([ymin, ymax, xmin, xmax])

        # get bbox coordinates in [ymin, ymax, xmin, xmax]
        t_y, t_x, t_h, t_w = bbox
        y = t_y * a_h + a_y
        x = t_x * a_w + a_x
        h = a_h * np.exp(t_h)
        w = a_w * np.exp(t_w)

        top_n_bboxes[i] = yxhw_to_yyxx([y, x, h, w])

    return top_n_categories, top_n_bboxes
