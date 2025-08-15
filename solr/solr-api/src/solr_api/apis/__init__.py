from flask_restx import Api

from .namespace_corpora import api as ns1
from .namespace_collections import api as ns2
from .namespace_models import api as ns3
from .namespace_queries import api as ns4

api = Api(
    title="TOVA's Solr API",
    version='1.0',
)

api.add_namespace(ns2, path='/collections')
api.add_namespace(ns1, path='/corpora')
#api.add_namespace(ns3, path='/models')
#api.add_namespace(ns4, path='/queries')