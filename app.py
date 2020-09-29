import glob
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, send_from_directory
import os
from metaheuritics import *
from tools import *
from fuzzy import *
from indexes import *
import matplotlib
from zipfile import ZipFile
import os
from os.path import basename
"""config = pdfkit.configuration(
    wkhtmltopdf=r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe")"""

matplotlib.use('Agg')
app = Flask(__name__, static_url_path='/static')


app.config['UPLOADED_PHOTOS_DEST'] = 'static\\img'


@app.route('/', methods=['GET', 'POST'])
def upload():
    files = glob.glob(r'static\\images/*')
    for items in files:
        os.remove(items)
    if request.method == 'POST' and 'image' in request.files:
        file = request.files["image"]
        print(file)
        file.save(os.path.join(
            app.config['UPLOADED_PHOTOS_DEST'], file.filename))
        return redirect('/traitment/' + file.filename)
    return render_template('index.html')


@app.route('/traitment/<imageName>', methods=['GET', 'POST'])
def traitment(imageName):
    plt.gray()
    path = "static\\img\\" + imageName
    files = glob.glob(r'static\\images/*')
    for items in files:
        os.remove(items)
    if request.method == "POST":
        print(request.form)
        x = metaheuristics(path=path, size=int(
            request.form["segments"]), m=int(request.form["fuzziness"]))
        if request.form["method"] == "Bat Optimization algorithm":
            centers, time = x.bat(N=int(request.form["population"]), GEN=int(request.form["generation"]), fmin=float(
                request.form["min_frequency"]), fmax=float(request.form["max_frequency"]), init_R=float(request.form["init_R"]), init_A=float(request.form["init_A"]), alpha=float(request.form["alpha"]), gamma=float(request.form["gamma"]))
        if request.form["method"] == "Particul Swarm Optimization":
            centers, time = x.pso(N=int(request.form["population"]), GEN=int(request.form["generation"]), vmin=float(
                request.form["min_velocity"]), vmax=float(request.form["max_velocity"]),
                constant1=float(request.form["constant1"]), constant2=float(request.form["constant2"]), maxw=float(request.form["wmax"]), minw=float(request.form["wmin"]))
        if request.form["method"] == "GrassHopper Optimization Algorithm":
            centers, time = x.gao(N=int(request.form["population"]), GEN=int(
                request.form["generation"]), f=float(request.form["f"]), l=float(request.form["l"]))
        if request.form["method"] == "Manual Centers":
            pass
        print(centers)
        f = FuzzyCMeans(n_clusters=int(request.form["segments"]), initial_centers=centers,
                        histogram=Histogram(path), m=int(request.form["fuzziness"]), max_iter=int(request.form["n"]))
        _centers, U, _time = f.compute()
        print(_centers)
        p = pc(U)
        c = ce(U, int(request.form["fuzziness"]))
        s = sc(Histogram(path), _centers, int(request.form["fuzziness"]))
        x = xb(Histogram(path), _centers, int(request.form["fuzziness"]))
        C = [str(int(i[0])) for i in _centers]
        image = imread(path)
        im = f.newImage(U, centers, image)
        plt.figure()
        plt.imshow(im, interpolation='nearest')
        plt.savefig(
            'static\images\\full.png', bbox_inches='tight')
        segs = []
        for i in centers:
            segs.append(numpy.where(im == i[0], i[0], 0))
        for i in range(len(segs)):
            plt.figure()
            plt.imshow(segs[i], interpolation='nearest')
            plt.savefig(
                'static\images\\' + C[i] + '.png', bbox_inches='tight')
        return render_template('results.html', partition_coefficient=p,
                               classification_entropy=c, xie_beni=x, time=time, centers=C)
    else:
        _histogram = HistogramV2(path)
        plt.figure()
        plt.plot(range(0, 256), _histogram)
        plt.savefig('static\img\\histogram.png', bbox_inches='tight')
        return render_template('traitment.html', imageName=imageName)


@app.route('/download')
def download():
    # create a ZipFile object
    with ZipFile('static/resutls.zip', 'w') as zipObj:
        # Iterate over all the files in directory
        for folderName, subfolders, filenames in os.walk("static\images"):
            for filename in filenames:
                # create complete filepath of file in directory
                filePath = os.path.join(folderName, filename)
                # Add file to zip
                zipObj.write(filePath, basename(filePath))
    return send_from_directory("static", "resutls.zip", as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
