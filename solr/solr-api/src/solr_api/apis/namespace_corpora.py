"""
This script defines a Flask RESTful namespace for managing corpora stored in Solr as collections. 

Author: Lorena Calvo-Bartolomé
Date: 27/03/2023
Modified: 15/08/2025 for TOVA project
"""

import io
import os
import pathlib
from flask_restx import Namespace, Resource, reqparse # type: ignore
from flask import request # type: ignore
from pydantic import ValidationError # type: ignore
from solr_api.core.clients.tova_solr_client import TOVASolrClient
from solr_api.models.corpus_requests import CorpusFileMeta, CorpusJSONRequest

# namespace for managing corpora
api = Namespace(
    'Corpora', description='Corpora-related operations (i.e., index/delete corpora))')

# Create Solr client
sc = TOVASolrClient(api.logger)

form_parser = api.parser()
form_parser.add_argument(
    "corpus_name",
    type=str,
    required=True,
    location="form",
    help="Name of the corpus/collection to index",
)
form_parser.add_argument(
    "file_path",
    type=str,
    required=True,
    location="form",
    help="Absolute (container-visible) path to the file on disk",
)

@api.route('/indexCorpus/')
class IndexCorpus(Resource):
    @api.doc(
        responses={
            200: 'Indexed',
            400: 'Validation error',
            404: 'File not found',
            500: 'Server error',
        },
    )
    @api.expect(form_parser, validate=True)
    def post(self):
        try:
            args = form_parser.parse_args()
            corpus_name: str = args["corpus_name"]
            file_path_str: str = args["file_path"]

            path = pathlib.Path(file_path_str)
            if not path.is_file():
                return {"error": f"File not found: {path}"}, 404

            sc.index_corpus(path.as_posix(), corpus_name,)
            return "", 200

        except Exception as e:
            # Keep it simple; surface a message for now
            return {"error": str(e)}, 500

@api.route('/deleteCorpus/<string:corpus_name>')
class DeleteCorpus(Resource):
    def delete(self, corpus_name: str):
        # @lcalvobartolome: simplify this when sc.delete_corpus has uniform return types
        try:
            result = sc.delete_corpus(corpus_name)

            if isinstance(result, tuple) and len(result) == 2:
                msg, code = result
                if 200 <= code < 300:
                    return "", code
                return {"error": msg if isinstance(msg, str) else "Failed to delete corpus."}, code

            if isinstance(result, int):
                if 200 <= result < 300:
                    return "", result
                return {"error": "Failed to delete corpus."}, result

            return "", 204

        except Exception as e:
            return {"error": str(e)}, 500
       
@api.route('/listCorpus/')
class ListCorpus(Resource):
    def get(self):
        try:
            corpus_lst, code = sc.list_corpus_collections()
            return {"corpora": corpus_lst}, code
        except Exception as e:
            return {"error": str(e)}, 500

# @api.route('/corpusModels/<string:corpus_col>')
# class CorpusModels(Resource):
#     def get(self, corpus_col: str):
#         try:
#             models, code = sc.get_corpus_models(corpus_col=corpus_col)
#             return {"models": models}, code
#         except Exception as e:
#             return {"error": str(e)}, 500
