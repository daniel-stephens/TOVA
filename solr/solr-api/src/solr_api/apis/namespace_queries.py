"""
This script defines a Flask RESTful namespace for managing Solr queries.

Author: Lorena Calvo-Bartolomé
Date: 13/04/2023
"""

from flask_restx import Namespace, Resource, reqparse
from solr_api.core.clients.tova_solr_client import TOVASolrClient

# ======================================================
# Define namespace for managing queries
# ======================================================
api = Namespace(
    'Queries', description='Specfic Solr queries.')

# ======================================================
# Namespace variables
# ======================================================
# Create Solr client
sc = TOVASolrClient(api.logger)

# Define parsers to take inputs from user
q1_parser = reqparse.RequestParser()
q1_parser.add_argument(
    'corpus_collection', help='Name of the corpus collection', required=True)
q1_parser.add_argument(
    'doc_id', help='ID of the document whose doc-topic distribution associated to a specific model is to be retrieved', required=True)
q1_parser.add_argument(
    'model_name', help='Name of the model reponsible for the creation of the doc-topic distribution to be retrieved', required=True)

q2_parser = reqparse.RequestParser()
q2_parser.add_argument(
    'corpus_collection', help='Name of the corpus collection', required=True)

q3_parser = reqparse.RequestParser()
q3_parser.add_argument(
    'collection', help='Name of the collection', required=True)

q4_parser = reqparse.RequestParser()
q4_parser.add_argument(
    'corpus_collection', help='Name of the corpus collection', required=True)
q4_parser.add_argument(
    'model_name', help='Name of the model reponsible for the creation of the doc-topic distribution', required=True)
q4_parser.add_argument(
    'topic_id', help='Topic whose proportion must be larger than a certain threshold', required=True)
q4_parser.add_argument(
    'threshold', help='Query threshold', required=True)
q4_parser.add_argument(
    'start', help='Specifies an offset (by default, 0) into the responses at which Solr should begin displaying content.', required=False)
q4_parser.add_argument(
    'rows', help='Controls how many rows of responses are displayed at a time (default value: maximum number of docs in the collection)', required=False)

q5_parser = reqparse.RequestParser()
q5_parser.add_argument(
    'corpus_collection', help='Name of the corpus collection', required=True)
q5_parser.add_argument(
    'model_name', help='Name of the model reponsible for the creation of the doc-topic distribution', required=True)
q5_parser.add_argument(
    'doc_id', help="ID of the document whose similarity is going to be checked against all other documents in 'corpus_collection'", required=True)
q5_parser.add_argument(
    'start', help='Specifies an offset (by default, 0) into the responses at which Solr should begin displaying content', required=False)
q5_parser.add_argument(
    'rows', help='Controls how many rows of responses are displayed at a time (default value: maximum number of docs in the collection)', required=False)

q6_parser = reqparse.RequestParser()
q6_parser.add_argument(
    'corpus_collection', help='Name of the corpus collection', required=True)
q6_parser.add_argument(
    'doc_id', help="ID of the document whose metadata is going to be retrieved'", required=True)

q7_parser = reqparse.RequestParser()
q7_parser.add_argument(
    'corpus_collection', help='Name of the corpus collection', required=True)
q7_parser.add_argument(
    'string', help="String to be search in the SearcheableField field'", required=True)
q7_parser.add_argument(
    'start', help='Specifies an offset (by default, 0) into the responses at which Solr should begin displaying content', required=False)
q7_parser.add_argument(
    'rows', help='Controls how many rows of responses are displayed at a time (default value: maximum number of docs in the collection)', required=False)

q8_parser = reqparse.RequestParser()
q8_parser.add_argument(
    'model_collection', help='Name of the model collection', required=True)
q8_parser.add_argument(
    'start', help='Specifies an offset (by default, 0) into the responses at which Solr should begin displaying content', required=False)
q8_parser.add_argument(
    'rows', help='Controls how many rows of responses are displayed at a time (default value: maximum number of docs in the collection)', required=False)

q9_parser = reqparse.RequestParser()
q9_parser.add_argument(
    'corpus_collection', help='Name of the corpus collection', required=True)
q9_parser.add_argument(
    'model_name', help='Name of the model reponsible for the creation of the doc-topic distribution', required=True)
q9_parser.add_argument(
    'topic_id', help="ID of the topic whose top documents according to 'model_name' are being searched", required=True)
q9_parser.add_argument(
    'start', help='Specifies an offset (by default, 0) into the responses at which Solr should begin displaying content', required=False)
q9_parser.add_argument(
    'rows', help='Controls how many rows of responses are displayed at a time (default value: maximum number of docs in the collection)', required=False)

q10_parser = reqparse.RequestParser()
q10_parser.add_argument(
    'model_collection', help='Name of the model collection', required=True)
q10_parser.add_argument(
    'start', help='Specifies an offset (by default, 0) into the responses at which Solr should begin displaying content', required=False)
q10_parser.add_argument(
    'rows', help='Controls how many rows of responses are displayed at a time (default value: maximum number of docs in the collection)', required=False)

q11_parser = reqparse.RequestParser()
q11_parser.add_argument(
    'model_collection', help='Name of the model collection', required=True)
q11_parser.add_argument(
    'topic_id', help='ID of the topic whose whose word-topic distribution is to be retrieved', required=False)

q12_parser = reqparse.RequestParser()
q12_parser.add_argument(
    'model_collection', help='Name of the model collection', required=True)
q12_parser.add_argument(
    'topic_id', help='ID of the topic whose whose word-topic distribution is to be retrieved', required=False)
q12_parser.add_argument(
    'start', help='Specifies an offset (by default, 0) into the responses at which Solr should begin displaying content', required=False)
q12_parser.add_argument(
    'rows', help='Controls how many rows of responses are displayed at a time (default value: maximum number of docs in the collection)', required=False)

q13_parser = reqparse.RequestParser()
q13_parser.add_argument(
    'corpus_collection', help='Name of the corpus collection', required=True)
q13_parser.add_argument(
    'model_name', help='Name of the model collection', required=True)
q13_parser.add_argument(
    'lower_limit', help='Lower percentage of semantic similarity to retrieve pairs of documents', required=True)
q13_parser.add_argument(
    'upper_limit', help='Upper percentage of semantic similarity to retrieve pairs of documents', required=True)
q13_parser.add_argument(
    'year', help='Publication year to be filtered by', required=False)
q13_parser.add_argument(
    'num_records', help='Controls how many pairs of documents are displayed at a time (default value: maximum number of docs in the collection)', required=False)

q14_parser = reqparse.RequestParser()
q14_parser.add_argument(
    'corpus_collection', help='Name of the corpus collection', required=True)
q14_parser.add_argument(
    'model_name', help='Name of the model reponsible for the creation of the doc-topic distribution', required=True)
q14_parser.add_argument(
    'text_to_infer', help="Text to be inferred", required=True)
q14_parser.add_argument(
    'start', help='Specifies an offset (by default, 0) into the responses at which Solr should begin displaying content', required=False)
q14_parser.add_argument(
    'rows', help='Controls how many rows of responses are displayed at a time (default value: maximum number of docs in the collection)', required=False)

q15_parser = reqparse.RequestParser()
q15_parser.add_argument(
    'corpus_collection', help='Name of the corpus collection', required=True)
q15_parser.add_argument(
    'doc_id', help='ID of the document whose whose doc-topic distribution associated to a specific model is to be retrieved', required=True)

q16_parser = reqparse.RequestParser()
q16_parser.add_argument(
    'corpus_collection', help='Name of the corpus collection', required=True)
q16_parser.add_argument(
    'model_name', help='Name of the model reponsible for the creation of the doc-topic distribution to be retrieved', required=True)
q16_parser.add_argument(
    'start', help='Specifies an offset (by default, 0) into the responses at which Solr should begin displaying content', required=False)
q16_parser.add_argument(
    'rows', help='Controls how many rows of responses are displayed at a time (default value: maximum number of docs in the collection)', required=False)

q17_parser = reqparse.RequestParser()
q17_parser.add_argument(
    'model_name', help='Name of the model to retrieve the topic-word distribution.', required=True)
q17_parser.add_argument(
    'tpc_id', help='ID of the specific topic to retrieve the topic-word distribution.', required=True)
q17_parser.add_argument(
    'word', help='Word of interest to retrieve its topic-word distribution in the specified topic. If the word is not present, the distribution is 0.', required=True)

q18_parser = reqparse.RequestParser()
q18_parser.add_argument(
    'corpus_name', help='Name of the corpus.', required=True)
q18_parser.add_argument(
    'ids', help='IDs of the documents separated by commas', required=True)
q18_parser.add_argument(
    'words', help='Words to get the BOW from, separated by commas', required=True)
q18_parser.add_argument(
    'start', help='Specifies an offset (by default, 0) into the responses at which Solr should begin displaying content', required=False)
q18_parser.add_argument(
    'rows', help='Controls how many rows of responses are displayed at a time (default value: maximum number of docs in the collection)', required=False)

q19_parser = reqparse.RequestParser()
q19_parser.add_argument(
    'model_collection', help='Name of the model collection', required=True)
q19_parser.add_argument(
    'user', help='Identifier/name of the user whose relevant topics are going to be retrieved', required=True)
q19_parser.add_argument(
    'start', help='Specifies an offset (by default, 0) into the responses at which Solr should begin displaying content', required=False)
q19_parser.add_argument(
    'rows', help='Controls how many rows of responses are displayed at a time (default value: maximum number of docs in the collection)', required=False)



@api.route('/getThetasDocById/')
class getThetasDocById(Resource):
    @api.doc(parser=q1_parser)
    def get(self):
        args = q1_parser.parse_args()
        corpus_collection = args['corpus_collection']
        doc_id = args['doc_id']
        model_name = args['model_name']

        try:
            return sc.do_Q1(corpus_col=corpus_collection,
                            doc_id=doc_id,
                            model_name=model_name)
        except Exception as e:
            return str(e), 500


@api.route('/getCorpusMetadataFields/')
class getCorpusMetadataFields(Resource):
    @api.doc(parser=q2_parser)
    def get(self):
        args = q2_parser.parse_args()
        corpus_collection = args['corpus_collection']
        try:
            return sc.do_Q2(corpus_col=corpus_collection)
        except Exception as e:
            return str(e), 500


@api.route('/getNrDocsColl/')
class getNrDocsColl(Resource):
    @api.doc(parser=q3_parser)
    def get(self):
        args = q3_parser.parse_args()
        collection = args['collection']
        try:
            return sc.do_Q3(col=collection)
        except Exception as e:
            return str(e), 500


@api.route('/getDocsWithThetasLargerThanThr/')
class getDocsWithThetasLargerThanThr(Resource):
    @api.doc(parser=q4_parser)
    def get(self):
        args = q4_parser.parse_args()
        corpus_collection = args['corpus_collection']
        model_name = args['model_name']
        topic_id = args['topic_id']
        threshold = args['threshold']
        start = args['start']
        rows = args['rows']

        try:
            return sc.do_Q4(corpus_col=corpus_collection,
                            model_name=model_name,
                            topic_id=topic_id,
                            thr=threshold,
                            start=start,
                            rows=rows)
        except Exception as e:
            return str(e), 500


@api.route('/getDocsWithHighSimWithDocByid/')
class getDocsWithHighSimWithDocByid(Resource):
    @api.doc(parser=q5_parser)
    def get(self):
        args = q5_parser.parse_args()
        corpus_collection = args['corpus_collection']
        model_name = args['model_name']
        doc_id = args['doc_id']
        start = args['start']
        rows = args['rows']

        try:
            return sc.do_Q5(corpus_col=corpus_collection,
                            model_name=model_name,
                            doc_id=doc_id,
                            start=start,
                            rows=rows)
        except Exception as e:
            return str(e), 500


@api.route('/getMetadataDocById/')
class getMetadataDocById(Resource):
    @api.doc(parser=q6_parser)
    def get(self):
        args = q6_parser.parse_args()
        corpus_collection = args['corpus_collection']
        doc_id = args['doc_id']

        try:
            return sc.do_Q6(corpus_col=corpus_collection,
                            doc_id=doc_id)
        except Exception as e:
            return str(e), 500


@api.route('/getDocsWithString/')
class getDocsWithString(Resource):
    @api.doc(parser=q7_parser)
    def get(self):
        args = q7_parser.parse_args()
        corpus_collection = args['corpus_collection']
        string = args['string']
        start = args['start']
        rows = args['rows']

        try:
            return sc.do_Q7(corpus_col=corpus_collection,
                            string=string,
                            start=start,
                            rows=rows)
        except Exception as e:
            return str(e), 500


@api.route('/getTopicsLabels/')
class getTopicsLabels(Resource):
    @api.doc(parser=q8_parser)
    def get(self):
        args = q8_parser.parse_args()
        model_collection = args['model_collection']
        start = args['start']
        rows = args['rows']

        try:
            return sc.do_Q8(model_col=model_collection,
                            start=start,
                            rows=rows)
        except Exception as e:
            return str(e), 500


@api.route('/getTopicTopDocs/')
class getTopicTopDocs(Resource):
    @api.doc(parser=q9_parser)
    def get(self):
        args = q9_parser.parse_args()
        corpus_collection = args['corpus_collection']
        model_name = args['model_name']
        topic_id = args['topic_id']
        start = args['start']
        rows = args['rows']

        try:
            return sc.do_Q9(corpus_col=corpus_collection,
                            model_name=model_name,
                            topic_id=topic_id,
                            start=start,
                            rows=rows)
        except Exception as e:
            return str(e), 500

@api.route('/getModelInfo/')
class getModelInfo(Resource):
    @api.doc(parser=q10_parser)
    def get(self):
        args = q10_parser.parse_args()
        model_collection = args['model_collection']
        start = args['start']
        rows = args['rows']
        try:
            return sc.do_Q10(model_col=model_collection,
                            start=start,
                            rows=rows,
                            only_id=False)
        except Exception as e:
            return str(e), 500


@api.route('/getBetasTopicById/')
class getBetasTopicById(Resource):
    @api.doc(parser=q11_parser)
    def get(self):
        args = q11_parser.parse_args()
        model_collection = args['model_collection']
        topic_id = args['topic_id']
        
        try:
            return sc.do_Q11(model_col=model_collection,
                            topic_id=topic_id)
        except Exception as e:
            return str(e), 500


@api.route('/getMostCorrelatedTopics/')
class getMostCorrelatedTopics(Resource):
    @api.doc(parser=q12_parser)
    def get(self):
        args = q12_parser.parse_args()
        model_collection = args['model_collection']
        topic_id = args['topic_id']
        start = args['start']
        rows = args['rows']

        try:
            return sc.do_Q12(model_col=model_collection,
                            topic_id=topic_id,
                            start=start,
                            rows=rows)
        except Exception as e:
            return str(e), 500

@api.route('/getPairsOfDocsWithHighSim/')
class getPairsOfDocsWithHighSim(Resource):
    @api.doc(parser=q13_parser)
    def get(self):
        args = q13_parser.parse_args()
        corpus_collection = args['corpus_collection']
        model_name = args['model_name']
        lower_limit = args['lower_limit']
        upper_limit = args['upper_limit']
        year = args['year']
        num_records = args['num_records']

        try:
            return sc.do_Q13(corpus_col=corpus_collection,
                            model_name=model_name,
                            lower_limit=lower_limit,
                            upper_limit=upper_limit,
                            year=year,
                            num_records=num_records)
        except Exception as e:
            return str(e), 500

@api.route('/getDocsSimilarToFreeText/')
class getDocsSimilarToFreeText(Resource):
    @api.doc(parser=q14_parser)
    def get(self):
        args = q14_parser.parse_args()
        corpus_collection = args['corpus_collection']
        model_name = args['model_name']
        text_to_infer = args['text_to_infer']
        start = args['start']
        rows = args['rows']

        try:
            return sc.do_Q14(corpus_col=corpus_collection,
                            model_name=model_name,
                            text_to_infer=text_to_infer,
                            start=start,
                            rows=rows)
        except Exception as e:
            return str(e), 500


@api.route('/getLemmasDocById/')
class getLemmasDocById(Resource):
    @api.doc(parser=q15_parser)
    def get(self):
        args = q15_parser.parse_args()
        corpus_collection = args['corpus_collection']
        doc_id = args['doc_id']

        try:
            return sc.do_Q15(corpus_col=corpus_collection,
                            doc_id=doc_id)
        except Exception as e:
            return str(e), 500


@api.route('/getThetasAndDateAllDocs/')
class getThetasAndDateAllDocs(Resource):
    @api.doc(parser=q16_parser)
    def get(self):
        args = q16_parser.parse_args()
        corpus_collection = args['corpus_collection']
        model_name = args['model_name']
        start = args['start']
        rows = args['rows']

        try:
            return sc.do_Q16(corpus_col=corpus_collection,
                            model_name=model_name,
                            start=start,
                            rows=rows)
        except Exception as e:
            return str(e), 500

@api.route('/getBetasByWordAndTopicId/')
class getBetasByWordAndTopicId(Resource):
    @api.doc(parser=q17_parser)
    def get(self):
        args = q17_parser.parse_args()
        model_name = args['model_name']
        tpc_id = args['tpc_id']
        word = args['word']

        try:
            return sc.do_Q17(model_name=model_name,
                            tpc_id=tpc_id,
                            word=word)
        except Exception as e:
            return str(e), 500
        
@api.route('/getBOWbyDocsIDs/')
class getBOWbyDocsIDs(Resource):
    @api.doc(parser=q18_parser)
    def get(self):
        args = q18_parser.parse_args()
        corpus_name = args['corpus_name']
        ids = args['ids']
        words = args['words']
        start = args['start']
        rows = args['rows']

        try:
            return sc.do_Q18(corpus_col=corpus_name,
                            ids=ids,
                            words=words,
                            start=start,
                            rows=rows,)
        except Exception as e:
            return str(e), 500
        
@api.route('/getUserRelevantTopics/')
class getUserRelevantTopics(Resource):
    @api.doc(parser=q19_parser)
    def get(self):
        args = q19_parser.parse_args()
        model_collection = args['model_collection']
        user = args['user']
        start = args['start']
        rows = args['rows']
        try:
            return sc.do_Q19(model_col=model_collection,
                            start=start,
                            rows=rows,
                            user=user)
        except Exception as e:
            return str(e), 500