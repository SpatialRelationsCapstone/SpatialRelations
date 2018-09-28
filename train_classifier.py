"""Train PredicateClassifier on top of YOLOv3 detector."""

import numpy as np

from skimage.io import imread

from keras_yolo3.yolo import YOLO
from relation_model.predicate_classifier import PredicateClassifier

import json


DATASET_DIR = "/floyd/input/sg_dataset/"


def train_classifier(annotations, detector, classifier,
                     epochs=5, print_every=1000, save_every=1000,
                     image_dir=DATASET_DIR + "sg_train_images/"):
    """Train predicate classification layer."""
    for epoch in range(epochs):
        print("Epoch {}".format(epoch + 1))

        iterations = 0
        for image_path, annotation in annotations.items():
            image = imread(image_dir + image_path)
            bbox_scaling_factor = np.tile(image.shape[:2], 2)

            image_input = np.expand_dims(image / 255, 0)
            detector_feed_dict = {detector.yolo_model.input: image_input}

            try:
                image_feature_map = detector.sess.run(
                    detector.yolo_model.get_layer("conv2d_13").output,
                    feed_dict=detector_feed_dict)
            except:
                continue

            for relation in annotation:
                target_index = [relation["predicate"]]

                subject_bbox = np.expand_dims(relation["subject"]["bbox"], 0)
                subject_bbox = subject_bbox / bbox_scaling_factor

                object_bbox = np.expand_dims(relation["object"]["bbox"], 0)
                object_bbox = object_bbox / bbox_scaling_factor

                classifier_feed_dict = {
                    classifier.image_feature_map: image_feature_map,
                    classifier.subject_bbox: subject_bbox,
                    classifier.object_bbox: object_bbox,
                    classifier.target_index: target_index}

                loss, _ = classifier.sess.run([classifier.loss,
                                               classifier.optimizer],
                                              feed_dict=classifier_feed_dict)

                if iterations % print_every == 0:
                    print("Iteration: {}, Loss: {}".format(iterations, loss))

                if iterations % save_every == 0:
                    classifier.save_model()

                iterations += 1

    classifier.save_model()


def _main():
    with open(DATASET_DIR + "annotations_train.json") as file:
        annotations = json.load(file)

    model_path = "/floyd/input/yolo_weights/trained_weights_final.h5"
    anchors_path = "keras_yolo3/model_data/tiny_yolo_anchors.txt"
    classes_path = "keras_yolo3/model_data/vrd_classes.txt"

    detector = YOLO(model_path=model_path,
                    anchors_path=anchors_path,
                    classes_path=classes_path)

    classifier = PredicateClassifier()

    train_classifier(annotations, detector, classifier)

if __name__ == '__main__':
    _main()
