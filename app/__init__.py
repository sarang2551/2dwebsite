from flask import Flask
from flask_bootstrap import Bootstrap4
from app.middleware import PrefixMiddleware

application = Flask(__name__)
application.config["SECRET_KEY"] = "IHaveACrushOnZovin"
Bootstrap4(application)
application.wsgi_app = PrefixMiddleware(application.wsgi_app, voc=True)
from app import routes