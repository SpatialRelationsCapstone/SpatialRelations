"""Flask Serving."""

from flask import Flask, request, jsonify
from werkzeug.exceptions import BadRequest
from werkzeug.utils import secure_filename

from skimage.io import imread

from evaluate import evaluate_relations
from keras_yolo3.yolo import YOLO
from relation_model.predicate_classifier import PredicateClassifier

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])
app = Flask(__name__)

model_path = "/floyd/input/weights/trained_weights_final.h5"
anchors_path = "keras_yolo3/model_data/tiny_yolo_anchors.txt"
classes_path = "keras_yolo3/model_data/vrd_classes.txt"

detector = YOLO(model_path=model_path,
                anchors_path=anchors_path,
                classes_path=classes_path)

classifier = PredicateClassifier()


@app.route('/', methods=["GET", "POST"])
def evaluate():
    """Take the input image and style transfer it."""
    # check if the post request has the file part
    input_file = request.files.get('file')
    if not input_file:
        return BadRequest("File not present in request")

    filename = secure_filename(input_file.filename)
    if filename == '':
        return BadRequest("File name is not present in request")
    if not _allowed_file(filename):
        return BadRequest("Invalid file type")

    image = imread(input_file)

    # Get checkpoint filename from la_muse
    annotations = evaluate_relations(image, detector, classifier)
    return jsonify(str(annotations))


def _allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == '__main__':
    app.run(host='0.0.0.0')
