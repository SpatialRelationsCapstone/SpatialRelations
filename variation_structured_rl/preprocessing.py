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


def get_anchor_coords(anchor):
    """Change anchor representation from yxhw to (ymin, ymax, xmin, xmax)."""
    y, x, h, w = anchor
    return y - h // 2, y + h // 2, x - w // 2, x + w // 2


def bbox_anchor_iou(anchor_coords, bbox_coords):
    """Calculate IoU between a bbox and an anchor.

    coords given as (ymin, ymax, xmin, xmax)
    """
    intersect_coords = [max(anchor_coords[0], bbox_coords[0]),
                        min(anchor_coords[1], bbox_coords[1]),
                        max(anchor_coords[2], bbox_coords[2]),
                        min(anchor_coords[3], bbox_coords[3])]

    intersect_area = calculate_area(intersect_coords)
    anchor_area = calculate_area(anchor_coords)
    bbox_area = calculate_area(bbox_coords)

    union_area = anchor_area + bbox_area - intersect_area

    return intersect_area / union_area


def ground_truth_offset(region_coords, bbox_corners):
    """Give the region-parameterized coordinates of a bounding box."""
    ya, xa, wa, ha = region_coords
    y, x, w, h = bbox_corners

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


def sample_ground_truth_targets(anchors,
                                annotation,
                                image_shape,
                                pos_threshold=0.7,
                                neg_threshold=0.3,
                                batch_size=256,
                                max_positive=128):
    """Generate minibatch of ground-truth anchor coordinates and labels."""
    # initialize ground truth labels to -1
    cls_target = np.zeros((anchors.shape[0:3])) - 1
    reg_target = np.zeros(anchors.shape) - 1

    # first gather bboxes (just for clarity of code)
    otypes = ["subject", "object"]
    bboxes = [rel[otype]["bbox"] for otype in otypes for rel in annotation]

    # overlapping examples from later bboxes will overwrite earlier ones
    for bbox in bboxes:
        max_iou = 0
        max_index = None
        max_anchor = None
        # iterate over each 4 tuple of anchor coords
        for index in np.ndindex(anchors.shape[0:3]):
            anchor = get_anchor_coords(anchors[index])

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

            iou = bbox_anchor_iou(anchor, bbox)
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
    """Wrap data processing functions."""
    image, annotation = resize_by_shorter(image, annotation)

    anchors = generate_anchors(image)

    cls_target, reg_target, loss_mask = sample_ground_truth_targets(
        anchors, annotation, image.shape)

    return image, anchors, cls_target, reg_target, loss_mask


