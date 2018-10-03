"""Detect objects and classify relations for an image."""
import numpy as np

import sys

from skimage.io import imread

from keras_yolo3.yolo import YOLO
from relation_model.predicate_classifier import PredicateClassifier


DATASET_DIR = "/floyd/input/weights/"


def create_annotation(predicate_id, bboxes, classes, subject_idx, object_idx):
    """Create an annotation in the style of the VRD dataset."""
    object_bbox = bboxes[object_idx, [0, 2, 1, 3]]
    subject_bbox = bboxes[subject_idx, [0, 2, 1, 3]]

    annotation = {"predicate": predicate_id,
                  "object": {"category": classes[object_idx],
                             "bbox": list(object_bbox)},
                  "subject": {"category": classes[subject_idx],
                              "bbox": list(subject_bbox)}}
    return annotation


def evaluate_relations(image, detector, classifier, threshold=0.12):
    """Detect objects, then report confidently classified relations."""
    image_input = np.expand_dims(image / 255, 0)
    image_shape = image.shape[0:2]

    feed_dict = {detector.yolo_model.input: image_input,
                 detector.input_image_shape: image_shape}

    try:
        feature_map, boxes, scores, classes = detector.sess.run(
            [detector.yolo_model.get_layer("conv2d_13").output,
             detector.boxes, detector.scores, detector.classes],
            feed_dict=feed_dict)
    except:
        return "YOLOv3 error, try resizing image."

    # exhaustively go through detected objects, looking for relations
    annotations = []
    for i in range(len(boxes)):
        for j in range(len(boxes)):
            if i == j:
                continue

            subject_bbox = boxes[i:i + 1] / np.tile(image_shape, 2)
            object_bbox = boxes[j:j + 1] / np.tile(image_shape, 2)

            feed_dict = {classifier.image_feature_map: feature_map,
                         classifier.subject_bbox: subject_bbox,
                         classifier.object_bbox: object_bbox}

            cls_output = classifier.sess.run(classifier.output,
                                             feed_dict=feed_dict)

            score = np.max(cls_output)
            relation = np.argmax(cls_output)

            if score > threshold:
                annotation = create_annotation(
                    relation, boxes.astype(int), classes, i, j)
                annotations.append(annotation)

    return annotations


def _main(image):
    model_path = DATASET_DIR + "trained_weights_final.h5"
    anchors_path = "keras_yolo3/model_data/tiny_yolo_anchors.txt"
    classes_path = "keras_yolo3/model_data/vrd_classes.txt"

    detector = YOLO(model_path=model_path,
                    anchors_path=anchors_path,
                    classes_path=classes_path)

    classifier = PredicateClassifier(save_path=DATASET_DIR + "pred_cls.ckpt")

    annotations = evaluate_relations(image, detector, classifier)
    print(annotations)

    return annotations


if __name__ == '__main__':
    image_path = sys.argv[1]
    image = imread(image_path)
    _main(image)
