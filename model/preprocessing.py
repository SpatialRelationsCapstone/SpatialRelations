"""Data preprocessing operations."""

import numpy as np

from skimage import resize


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


def generate_anchor_coordinates(image,
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
    anchors = anchors.reshape(*y_ctr.shape[:2], 4 * len(sides) * len(ratios))
    return anchors
