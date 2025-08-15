"""
This script defines a Flask RESTful namespace for managing models stored in Solr as collections. 

Author: Lorena Calvo-Bartolomé
Date: 27/03/2023
"""

from flask_restx import Namespace, Resource, reqparse
from solr_api.core.clients.tova_solr_client import TOVASolrClient

# ======================================================
# Define namespace for managing models
# ======================================================
api = Namespace(
    'Models', description='Models-related operations (i.e., index/delete models))')

# ======================================================
# Namespace variables
# ======================================================
# Create Solr client
sc = TOVASolrClient(api.logger)

# Define parser to take inputs from user
parser = reqparse.RequestParser()
parser.add_argument(
    'model_name', help="Name of the model to index/index. You should specify the name of the folder in which topic model information is stored.")

parser_add_rel = reqparse.RequestParser()
parser_add_rel.add_argument(
    'model_name', help="Name of the model to which the relevant topic will be added")
parser_add_rel.add_argument(
    'topic_id', help="Topic id of the relevant topic to be added")
parser_add_rel.add_argument(
    'user', help="User who is adding the relevant topic")

parser_del_rel = reqparse.RequestParser()
parser_del_rel.add_argument(
    'model_name', help="Name of the model to which the relevant topic will be removed")
parser_del_rel.add_argument(
    'topic_id', help="Topic id of the relevant topic to be removed")
parser_del_rel.add_argument(
    'user', help="User who is deleting the relevant topic")

@api.route('/indexModel/')
class IndexModel(Resource):
    @api.doc(parser=parser)
    def post(self):
        args = parser.parse_args()
        model_name = args['model_name']
        try:
            sc.index_model(model_name)
            return '', 200
        except Exception as e:
            return str(e), 500


@api.route('/deleteModel/')
class DeleteModel(Resource):
    @api.doc(parser=parser)
    def post(self):
        args = parser.parse_args()
        model_name = args['model_name']
        try:
            sc.delete_model(model_name)
            return '', 200
        except Exception as e:
            return str(e), 500

@api.route('/listAllModels/')
class ListAllModels(Resource):
    def get(self):
        try:
            models_lst, code = sc.list_model_collections()
            return models_lst, code
        except Exception as e:
            return str(e), 500
        

@api.route('/addRelevantTpcForUser/')
class addRelevantTpcForUser(Resource):
    @api.doc(parser=parser_add_rel)
    def post(self):
        args = parser_add_rel.parse_args()
        model_name = args['model_name']
        tpc_id = args['topic_id']
        user = args['user']
        try:
            sc.modify_relevant_tpc(
                model_col=model_name,
                topic_id=tpc_id,
                user=user,
                action="add-distinct"
            )
            return '', 200
        except Exception as e:
            return str(e), 500

@api.route('/removeRelevantTpcForUser/')
class removeRelevantTpcForUser(Resource):
    @api.doc(parser=parser_del_rel)
    def post(self):
        args = parser_del_rel.parse_args()
        model_name = args['model_name']
        tpc_id = args['topic_id']
        user = args['user']
        try:
            sc.modify_relevant_tpc(
                model_col=model_name,
                topic_id=tpc_id,
                user=user,
                action="remove"
            )
            return '', 200
        except Exception as e:
            return str(e), 500