"""
This script defines a Flask RESTful namespace for managing Solr collections. 

Author: Lorena Calvo-Bartolomé
Date: 27/03/2023
"""

from flask_restx import Namespace, Resource, reqparse # type: ignore
from solr_api.core.clients.tova_solr_client import TOVASolrClient

# ======================================================
# Define namespace for managing collections
# ======================================================
api = Namespace('Collections',
                description='Generic Solr-related operations (collections creation and deletion, queries, etc.)')

# ======================================================
# Namespace variables
# ======================================================
# Create Solr client
sc = TOVASolrClient(api.logger)

# Define parsers to take inputs from user
parser = reqparse.RequestParser()
parser.add_argument('collection', help='Collection name')

query_parser = reqparse.RequestParser()
query_parser.add_argument(
    'collection', help='Collection name on which you want to execute the query. This parameter is mandatory', required=True)
query_parser.add_argument(
    'q', help="Defines a query using standard query syntax. This parameter is mandatory", required=True)
query_parser.add_argument(
    'q.op', help="Specifies the default operator for query expressions, overriding the default operator specified in the Schema. Possible values are 'AND' or 'OR'.")
query_parser.add_argument(
    'fq', help="Applies a filter query to the search results")
query_parser.add_argument(
    'sort', help="Sorts the response to a query in either ascending or descending order based on the response's score or another specified characteristic")
query_parser.add_argument(
    'start', help="Specifies an offset (by default, 0) into the responses at which Solr should begin displaying content")
query_parser.add_argument(
    'rows', help="Controls how many rows of responses are displayed at a time (default value: 10)")
query_parser.add_argument(
    'fl', help="Limits the information included in a query response to a specified list of fields. The fields need to either be stored='true' or docValues='true'")
query_parser.add_argument(
    'df', help="Specifies a default field, overriding the definition of a default field in the Schema.")

# ======================================================
# Methods
# ======================================================
@api.route('/createCollection/')
class CreateCollection(Resource):
    @api.doc(parser=parser)
    def post(self):
        args = parser.parse_args()
        collection = args['collection']
        try:
            corpus, err = sc.create_collection(col_name=collection)
            if err == 409:
                return f"Collection {collection} already exists.", err
            else:
                return corpus, err
        except Exception as e:
            return str(e), 500

@api.route('/deleteCollection/')
class DeleteCollection(Resource):
    @api.doc(parser=parser)
    def post(self):
        args = parser.parse_args()
        collection = args['collection']
        try:
            sc.delete_collection(col_name=collection)
            return f"Collection {collection} was deleted.", 200
        except Exception as e:
            return str(e), 500


@api.route('/listCollections/')
class ListCollections(Resource):
    def get(self):
        try:
            return sc.list_collections()
        except Exception as e:
            return str(e), 500

@api.route('/query/')
class Query(Resource):
    @api.doc(parser=query_parser)
    def get(self):
        args = query_parser.parse_args()
        collection = args['collection']
        q = args['q']
        query_values = {
            'q_op': args['q.op'],
            'fq': args['fq'],
            'sort': args['sort'],
            'start': args['start'],
            'rows': args['rows'],
            'fl': args['fl'],
            'df': args['df']
        }

        if q is None:
            return "Query is mandatory", 400

        if collection is None:
            return "Collection is mandatory", 400

        # Remove all key-value pairs with value of None
        query_values = {k: v for k, v in query_values.items() if v is not None}

        # Execute query
        try:
            code, results = sc.execute_query(
                q=q, col_name=collection, **query_values)
            return results.docs, code
        except Exception as e:
            return str(e), 500