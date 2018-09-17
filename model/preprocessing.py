"""Data preprocessing operations."""

from skimage import resize


def resize_by_shorter(image, annotation, shorter_length=600, mode="reflect"):
    """Resize an image such that its shorter dimension is `shorter_length`."""
    height, width, _ = image.shape

    shorter = min(height, width)
    longer = max(height, width)

    r = shorter_length / shorter
    longer_length = int(longer * r)

    if height > width:
        resized = resize(image, (longer_length, shorter_length), mode=mode)
    else:
        resized = resize(image, (shorter_length, longer_length), mode=mode)

    for rel in annotation:
        rel["object"]["bbox"] = [int(n * r) for n in rel["object"]["bbox"]]
        rel["subject"]["bbox"] = [int(n * r) for n in rel["subject"]["bbox"]]

    return resized, annotation
