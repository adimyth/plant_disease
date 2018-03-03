import os
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename
from flask import *
from flask import Flask, request, redirect, url_for
import predict

UPLOAD_FOLDER = './upload_folder'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET','POST'])
def upload_file():
    # for i in request.__dict__:
    #     print(i)
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # predict.prediction(os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return render_template("index.html")

@app.route('/results')
def training_results():
    return render_template("training_plots.html")

@app.route('/about')
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.debug = True
    app.run()