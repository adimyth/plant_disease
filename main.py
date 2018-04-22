import os
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename
from flask import *
from flask import Flask, request, redirect, url_for
import predict
import glob
from shutil import copyfile

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'upload_folder')
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
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            predict.prediction()
            copyfile("main.html", "templates/main.html")
    return render_template("index.html")

@app.route('/clean')
def clean():
    print("[INFO] cleaning upload folder...")
    files = glob.glob('upload_folder/*')
    for f in files:
        os.remove(f)
    return render_template("index.html")

@app.route('/results')
def results():
    return render_template("main.html")

@app.route('/about')
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.debug = True
    app.run()