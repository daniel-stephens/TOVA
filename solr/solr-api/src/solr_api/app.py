"""Main application entry point
"""
from solr_api.apis import api
from flask import Flask # type: ignore

# Create Flask app
app = Flask(__name__)
# Deactivate the default mask parameter
app.config["RESTX_MASK_SWAGGER"] = False
api.init_app(app)