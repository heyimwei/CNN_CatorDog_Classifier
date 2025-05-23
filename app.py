from flask import Flask, request, render_template, redirect, url_for
from inference import Classifier
from werkzeug.exceptions import RequestEntityTooLarge
import os
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2MB
app.config['UPLOAD_FOLDER'] = 'static/uploads'

clf = Classifier()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', result='錯誤：未選擇檔案')

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', result='錯誤：尚未選擇任何檔案')

    try:
        # 儲存圖片
        filename = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4().hex}_{filename}"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
        file.save(save_path)

        # 執行預測
        with open(save_path, 'rb') as f:
            result = clf.predict(f)

        image_url = url_for('static', filename='uploads/' + unique_name)
        return render_template('index.html', result=f'分類結果：{result}', image_url=image_url)

    except ValueError as ve:
        return render_template('index.html', result=f'錯誤：{ve}')
    except Exception as e:
        return render_template('index.html', result='錯誤：無法解析圖片或伺服器異常')

@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    return render_template('index.html', result='錯誤：檔案過大（上限 2MB）'), 413

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)

