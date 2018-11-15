import re

from flask import Flask, request
from flask.json import jsonify

from engine import Engine
from utils import load_base64

app = Flask("hashnet server")
engine = Engine()


@app.route("/test")
def test():
    return "The Hashnet server is up and running like a champ."


@app.route("/hashnet/match", methods=['POST'])
def match():
    image_base64 = re.sub('^data:image/.+;base64,', '', request.form['data'])
    target_image = load_base64(image_base64)
    if target_image is None:
        return jsonify({
            "success": False,
            "message": "图片太小看不清, 请确保短边≥160px"
        })
    result, similarity = engine.search(target_image, page_size=500)
    if result is None:
        return jsonify({
            "success": False,
            "message": "库中无可用图片"
        })
    return jsonify({
        "similarity": similarity,
        "result": result.serialize(),
        "success": True
    })
