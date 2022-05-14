from flask import Flask
from flask import render_template, request, redirect, url_for
import os
import numpy as np
from model import model, predict_labels, load_image, tile_images
import flash
from werkzeug.utils import secure_filename, send_from_directory

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



# create updoad image button
@app.route('/', methods=['GET', 'POST'])
def image_preprocess():
    if request.method == 'POST':
        # check if the post request has the file part
       
        # files = request.files['file']
        files = request.files.getlist('files[]')
        file_list = []
        for file in files:
            # if user does not select file, browser also
            # submit an empty part without filename             

            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                file_list.append('static/uploads/'+filename)
        print(file_list)
        file = tile_images(file_list)
        # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('uploaded_file',
                                    filename=filename))
    return render_template('img_preprocessing.html')                                    
                                

@app.route('/uploads/<filename>')
def uploaded_file(filename):


    img = load_image('static/uploads/'+filename)
    txt = predict_labels(model, img)
    filepath = 'uploads/'+filename
    # render template result with image     
    return render_template('result.html', image_name=filepath, result=txt)





# add progress bar page
@app.route('/progress_bar')
def progress_bar():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Progress Bar</title>
    </head>
    <body>
        <h1>Progress Bar</h1>
        <div class="progress">
            <div class="progress-bar progress-bar-striped progress-bar-animated" role="progressbar" aria-valuenow="75" aria-valuemin="0" aria-valuemax="100" style="width: 75%"></div>
        </div>
    </body>
    </html>
    '''




# start flask app
if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)


