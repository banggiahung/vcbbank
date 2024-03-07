from flask import Flask
from flask_cors import CORS, cross_origin
from flask import request
from flask import jsonify

import os
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from collections import Counter

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time
# Khởi tạo Flask Server Backend 
app = Flask(__name__)
# Xác nhận Flask CORS 
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

characters = ['1', '2', '3', '4', '5', '6', '7', '8', '9']

batch_size = 16
# Chiều dài và chiều rộng của cái captcha 
img_width = 155
img_height = 50
downsample_factor = 4

max_length = 15
char_to_num = layers.StringLookup(vocabulary=list(characters), mask_token=None)
num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)






# Tạo class CTCLayer | DichVuDark.Vn
class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    


# Tiến hành nhập model giải captcha | DichVuDark.Vn
model = keras.models.load_model("vcb_model.h5", custom_objects={"CTCLayer": CTCLayer})
# Lấy dạng layer của hình ảnh | DichVuDark.Vn 
prediction_model = keras.models.Model(
    model.get_layer(name="image").input, model.get_layer(name="dense2").output
)

# Đọc ảnh base64 và mã hóa | DichVuDark.Vn
def encode_base64x(base64):
    img = tf.io.decode_base64(base64)
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [img_height, img_width])
    img = tf.transpose(img, perm=[1, 0, 2])
    return {"image": img}
# Tiện ích để giải mã đầu ra của mạng nơ ron | DichVuDark.Vn
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_length
    ]
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text





# Tạo api để sử dụng, thêm route cho nó sang chảnh | DichVuDark.Vn
@app.route("/quannguyen/captcha/vcb", methods=["POST"])
@cross_origin(origin='*')
def mb():
    content = request.json
    start_time = time.time()
    imgstring = content['imgbase64']
    image_encode = encode_base64x(imgstring.replace("+", "-").replace("/", "_"))["image"]
    listImage = np.array([image_encode])
    preds = prediction_model.predict(listImage)
    pred_texts = decode_batch_predictions(preds)
    captcha = pred_texts[0].replace('[UNK]', '').replace('-', '')
    response = jsonify(status = "success",captcha = captcha)

    return response


# Chạy server, chọn port tùy nhu cầu | DichVuDark.Vn
if __name__ == '__main__':
    app.run(host='0.0.0.0', port='1406')
