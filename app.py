from flask import Flask
import pandas as pd
# from segmentation_model import get_building_img
from flask import render_template, request, redirect, url_for
import os
import numpy as np
from model import catboost, load_image, tile_images
from model import model, predict_labels
import flash
import torch
from werkzeug.utils import secure_filename


UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

df = pd.read_csv('houses.csv')
torch.cuda.empty_cache()



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



# create updoad image button
@app.route('/', methods=['GET', 'POST'])
def image_preprocess():
    if request.method == 'POST':
        rooms = float(request.form['rooms'])
        baths = float(request.form['baths'])
        square = float(request.form['square'])
        zip = float(request.form['zip'])
        answers = [rooms, baths, square, zip]
        answers = [float(i) for i in answers]
        print(answers)
        # get column value of first row from df with condition

        true_price = (df['price'][(df['rooms'] == rooms) & (df['baths'] == baths) & (df['square'] == square) & (df['post'] == zip)]).values
        price = catboost.predict(answers)
        if len(true_price) != 0:
        # if true_price != np.nan:
            percent = (round(abs(true_price[0] - price) / true_price[0] * 100, 2))
            print(percent)
           
        else:
            percent = 'New house'
        price = round(price, 2)
        
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
                                    filename=filename, price=price, percent=percent))
    return render_template('img_preprocessing.html')                                    
                                

@app.route('/uploads/<filename>_<price>_<percent>')
def uploaded_file(filename, price, percent):
    print(percent)


    img = load_image('static/uploads/'+filename)
    
    txt = predict_labels(model, img)

    filepath = 'uploads/'+filename
    filename = filename.replace('kitchen', 'frontal')
    building_path = 'building/'+filename


    return render_template('result.html', image_name=filepath, build_name=building_path, result=txt, price=price, percent=percent)



# start flask app
if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)


