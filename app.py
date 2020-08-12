import glob
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect
import os
from metaheuritics import *
from tools import *
from fuzzy import *
from indexes import *
import matplotlib
matplotlib.use('Agg')
app = Flask(__name__, static_url_path='/static')


app.config['UPLOADED_PHOTOS_DEST'] = 'static\\img'


@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' and 'image' in request.files:
        file = request.files["image"]
        print(file)
        file.save(os.path.join(
            app.config['UPLOADED_PHOTOS_DEST'], file.filename))
        return redirect('/traitment/' + file.filename)
    return render_template('index.html')


@app.route('/traitment/<imageName>', methods=['GET', 'POST'])
def traitment(imageName):
    if request.method == "POST":
        path = "static\\img\\" + imageName
        x = metaheuristics(path=path, size=int(
            request.form["segments"]), m=int(request.form["fuzziness"]))
        if request.form["method"] == "Bat Optimization algorithm":
            centers, time = x.bat(N=int(request.form["population"]), GEN=int(request.form["generation"]), fmin=float(
                request.form["min_frequency"]), fmax=float(request.form["max_frequency"]))
        if request.form["method"] == "Particul Swarm Optimization":
            centers, time = x.pso(N=int(request.form["population"]), GEN=int(request.form["generation"]), vmin=float(
                request.form["min_velocity"]), vmax=float(request.form["max_velocity"]),
                constant1=float(request.form["constant1"]), constant2=float(request.form["constant2"]), weight=float(request.form["weight"]))
        if request.form["method"] == "GrassHopper Optimization Algorithm":
            centers = x.gao(N=int(request.form["population"]), GEN=int(
                request.form["generation"]), f=float(request.form["f"]), l=float(request.form["l"]))
        if request.form["method"] == "Manual Centers":
            pass
        f = FuzzyCMeans(n_clusters=int(request.form["segments"]), initial_centers=centers,
                        histogram=Histogram(path), m=int(request.form["fuzziness"]), max_iter=2000)
        _centers, U = f.compute()
        p = pc(U)
        c = ce(U, int(request.form["fuzziness"]))
        s = sc(Histogram(path), _centers, int(request.form["fuzziness"]))
        x = xb(Histogram(path), _centers, int(request.form["fuzziness"]))
        C = [str(int(i[0])) for i in _centers]
        image = imread(path)
        im = f.newImage(U, centers, image)
        segs = []
        for i in centers:
            segs.append(numpy.where(im == i[0], i[0], 0))
        for i in range(len(segs)):
            plt.imshow(segs[i], cmap='gray')
            plt.savefig(
                'static\\' + C[i] + '.png', bbox_inches='tight')
        return render_template('results.html', partition_coefficient=p, classification_entropy=c, xie_beni=x, subarea_coefficient=s, centers=C)
    return render_template('traitment.html', imageName=imageName)


if __name__ == '__main__':
    app.run(debug=True)
