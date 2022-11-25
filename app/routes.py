from app import application
from flask import render_template, request
from app.serverlogic import make_prediction, make_comparison
from app.forms import Predict_form
from markupsafe import Markup


@application.route('/')
@application.route('/about')
def index():
    return render_template('about.html',title="About our project")

@application.route("/predict",methods = ['GET','POST'])
def predict_page():
    # handle both get and post requests
    predict_form = Predict_form()
    if request.method == "POST":
        if predict_form.validate_on_submit():
            # request form structure 
            # form keys = ["population","arable_land"]
            results = dict(request.form)
            current_imports = float(results['current_imports']) # for comparison
            result = make_prediction(results)
            results['comparison'],results["less"] = make_comparison(current_imports,result)
            results['prediction'] = result
            
            return render_template("predict.html", results = results, form=predict_form)
    else:
        return render_template("predict.html",form=predict_form)

@application.route("/graph/cri_score")
def cri_score():
    return render_template("cri_score.html")

@application.route("/graph/population")
def mean_population():
    return render_template("mean_population.html")

@application.route("/graphs/local_agriculture")
def local_agriculture():
    return render_template("local_agriculture.html")

@application.route("/graphs/arable_land")
def mean_arable():
    return render_template("mean_arable.html")

@application.route("/graphs/credit_agriculture")
def credit_agriculture():
    return render_template("credit_agriculture_millions.html")

@application.route("/graphs/political")
def mean_political():
    return render_template("mean_political.html")