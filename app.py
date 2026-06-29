import os
import cv2
from pathlib import Path

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from onnx_infer import infer_image

# 创建Flask对象
# __name__ 表示当前这个 python 文件的模块名
# Flask 会根据它找到 templates 和 static 目录

app = Flask(__name__)

UPLOAD_FOLDER = Path("static/uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
RESULT_FOLDER = Path("static/results")
os.makedirs(RESULT_FOLDER, exist_ok=True)

# 定义首页路由
# 当浏览器访问 http://127.0.0.1:5000/ 时，会执行 index() 函数
@app.route("/", methods=["GET", "POST"])
def index():
    image_path = None
    mask_path = None

    if request.method == "POST":
        file = request.files.get("image")

        if file is not None and file.filename != "":
            filename = secure_filename(file.filename)

            save_path = UPLOAD_FOLDER / filename
            file.save(save_path)

            image_path = str(save_path)

            original, pred_mask = infer_image(str(save_path))

            result_filename = "mask_" + filename
            result_path = RESULT_FOLDER / result_filename

            cv2.imwrite(str(result_path), pred_mask)

            mask_path = str(result_path)

    return render_template(
        "index.html",
        image_path=image_path,
        mask_path=mask_path
    )

# 程序入口
# 只有直接运行 python app.py 时，才会启动 Flask 服务
if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
