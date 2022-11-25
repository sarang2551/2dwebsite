from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, IntegerField
from wtforms.validators import InputRequired,NumberRange

class Predict_form(FlaskForm):
    mean_population = IntegerField("Population (Number of people in your country)",validators=[InputRequired(),NumberRange(min=0,max=3*10**9)]) # population of a single country can't be more than 3 billiob
    mean_arable = IntegerField("Arable Land (Percentage of arable land in your country)",validators=[InputRequired(),NumberRange(min=0,max=100)]) # this is a percentage
    credit_agriculture_millions = IntegerField("Agriculture to GDP (Actual value is $US from the agricultural sector to GDP)",validators=[InputRequired(),NumberRange(min=0)])
    cri_score = IntegerField("Crisis index ( crisis index according to U.N )",validators=[InputRequired(),NumberRange(min=10,max=150)])
    local_agriculture = IntegerField("Local agriculture ( Gross production index number )",validators=[InputRequired(),NumberRange(min=0)])
    mean_political = IntegerField("Political stability index ( Political stability accoring to U.N )",validators=[InputRequired(),NumberRange(min=-5,max=5)])
    current_imports = IntegerField("Current food imports ( Food imports index ) ",validators=[InputRequired(),NumberRange(min=0)]) # index value
    submitButton = SubmitField(" Predict ")