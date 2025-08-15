"""
This module defines a class with the EWB-specific queries used to interact with Solr.


Author: Lorena Calvo-Bartolomé
Date: 19/04/2023
"""


class Queries(object):

    def __init__(self) -> None:

        # ================================================================
        # # Q1: getThetasDocById  ##################################################################
        # # Get document-topic distribution of a selected document in a
        # # corpus collection
        # http://localhost:8983/solr/{col}/select?fl=doctpc_{model}&q=id:{id}
        # ================================================================
        self.Q1 = {
            'q': 'id:{}',
            'fl': 'doctpc_{}',
        }

        # ================================================================
        # # Q2: getCorpusMetadataFields  ##################################################################
        # # Get the name of the metadata fields available for
        # a specific corpus collection (not all corpus have
        # the same metadata available)
        # http://localhost:8983/solr/#/Corpora/query?q=corpus_name:Cordis&q.op=OR&indent=true&fl=fields&useParams=
        # ================================================================
        self.Q2 = {
            'q': 'corpus_name:{}',
            'fl': 'fields',
        }

        # ================================================================
        # # Q3: getNrDocsColl ##################################################################
        # # Get number of documents in a collection
        # http://localhost:8983/solr/{col}/select?q=*:*&wt=json&rows=0
        # ================================================================
        self.Q3 = {
            'q': '*:*',
            'rows': '0',
        }

        # ================================================================
        # # Q4: GetDocsWithThetasLargerThanThr ##################################################################
        # # Get documents that have a proportion of a certain topic larger
        # # than a threshold
        # q={!payload_check f=doctpc_{tpc} payloads="{thr}" op="gte"}t{tpc}
        # ================================================================
        self.Q4 = {
            'q': "{{!payload_check f=doctpc_{} payloads='{}' op='gte'}}t{}",
            'start': '{}',
            'rows': '{}',
            'fl': "id,doctpc_{}"
        }

        # ================================================================
        # # Q5: getDocsWithHighSimWithDocByid
        # ################################################################
        # # Retrieve documents that have a high semantic relationship with
        # # a selected document
        # ---------------------------------------------------------------
        # Previous steps:
        # ---------------------------------------------------------------
        # 1. Get thetas of selected documents
        # 2. Parse thetas in Q1
        # 3. Execute Q4
        # ================================================================
        self.Q5 = {
            'q': "{{!vp f=doctpc_{} vector=\"{}\"}}",
            'fl': "id,score",
            'start': '{}',
            'rows': '{}'
        }

        # ================================================================
        # # Q6: getMetadataDocById
        # ################################################################
        # # Get metadata of a selected document in a corpus collection
        # ---------------------------------------------------------------
        # Previous steps:
        # ---------------------------------------------------------------
        # 1. Get metadata fields of that corpus collection with Q2
        # 2. Parse metadata in Q6
        # 3. Execute Q6
        # ================================================================
        self.Q6 = {
            'q': 'id:{}',
            'fl': '{}'
        }

        # ================================================================
        # # Q7: getDocsWithString
        # ################################################################
        # # Given a corpus collection, it retrieves the ids of the documents whose title contains such a string
        # http://localhost:8983/solr/#/{collection}/query?q=title:{string}&q.op=OR&indent=true&useParams=
        # ================================================================
        self.Q7 = {
            'q': '{}:{}',
            'fl': 'id',
            'start': '{}',
            'rows': '{}'
        }

        # ================================================================
        # # Q8: getTopicsLabels
        # ################################################################
        # # Get the label associated to each of the topics in a given model
        # http://localhost:8983/solr/{model}/select?fl=id%2C%20tpc_labels&indent=true&q.op=OR&q=*%3A*&useParams=
        # ================================================================
        self.Q8 = {
            'q': '*:*',
            'fl': 'id,tpc_labels',
            'start': '{}',
            'rows': '{}'
        }

        # ================================================================
        # # Q9: getTopicTopDocs
        # ################################################################
        # # Get the top documents for a given topic in a model collection
        # http://localhost:8983/solr/cordis/select?indent=true&q.op=OR&q=%7B!term%20f%3D{model}%7Dt{topic_id}&useParams=
        # http://localhost:8983/solr/#/{corpus_collection}/query?q=*:*&q.op=OR&indent=true&fl=doctpc_{model_name},%20nwords_per_doc&sort=payload(doctpc_{model_name},t{topic_id})%20desc,%20nwords_per_doc%20desc&useParams=
        # ================================================================
        """
        self.Q9 = {
            'q': '*:*',
            'sort': 'payload(doctpc_{},t{}) desc, nwords_per_doc desc',
            'fl': 'payload(doctpc_{},t{}), nwords_per_doc, id',
            'start': '{}',
            'rows': '{}'
        }#doctpc_{}
        """
        self.Q9 = {
            'q': '*:*',
            'sort': 'payload(s3_{},t{}) desc, nwords_per_doc desc',
            'fl': 'payload(s3_{},t{}), nwords_per_doc, id',
            'start': '{}',
            'rows': '{}'
        }#doctpc_{}

        # ================================================================
        # # Q10: getModelInfo
        # ################################################################
        # # Get the information (chemical description, label, statistics,
        # top docs, etc.) associated to each topic in a model collection
        # ================================================================
        self.Q10 = {
            'q': '*:*',
            'fl': 'id,alphas,top_words_betas,topic_entropy,topic_coherence,ndocs_active,tpc_descriptions,tpc_labels,coords',
            'start': '{}',
            'rows': '{}'
        }

        # ================================================================
        # # Q11: getBetasTopicById  ##################################################################
        # # Get word distribution of a selected topic in a
        # # model collection
        # http://localhost:8983/solr/{col}/select?fl=betas&q=id:t{id}
        # ================================================================
        self.Q11 = {
            'q': 'id:t{}',
            'fl': 'betas',
        }

        # ================================================================
        # # Q12: getMostCorrelatedTopics
        # ################################################################
        # # Get the most correlated topics to a given one in a selected
        # model
        # ================================================================
        self.Q12 = {
            'q': "{{!vp f=betas vector=\"{}\"}}",
            'fl': "id,score",
            'start': '{}',
            'rows': '{}'
        }

        # ================================================================
        # # Q13: getPairsOfDocsWithHighSim
        # ################################################################
        # # Get pairs of documents with a semantic similarity larger than a threshold
        # ================================================================
        self.Q13 = {
            'q': "{{!vs f=sim_{} vector=\"{},{}\"}} & date:[{}-01-01T00:00:00Z TO {}-12-31T23:59:59Z]",
            'q_no_date': "{{!vs f=sim_{} vector=\"{},{}\"}}",
            'fl': "id, sim_{}, score",
            'start': '{}',
            'rows': '{}'
        }

        # ================================================================
        # # Q14: getDocsSimilarToFreeText
        # ################################################################
        # # Get documents that are semantically similar to a free text
        # according to a given model
        # ================================================================
        self.Q14 = self.Q5

        # ================================================================
        # # Q15: getLemmasDocById  ##################################################################
        # # Get lemmas of a selected document in a corpus collection
        # http://localhost:8983/solr/{col}/select?fl=lemmas&q=id:{id}
        # ================================================================
        self.Q15 = {
            'q': 'id:{}',
            'fl': 'lemmas',
        }

        # ================================================================
        # # Q16: getThetasAndDateAllDocs  ##################################################################
        # # Get the document-topic representation and date of all documents in a corpus collection and selected model. Note that for documents with no document-topic representation, only the date field is returned
        # http://localhost:8983/solr/{col}/query?q=*:*&q.op=OR&indent=true&fl=doctpc_{model},date&rows=1000&useParams=
        # ================================================================
        self.Q16 = {
            'q': '*:*',
            'fl': 'id,date,doctpc_{}',
            'start': '{}',
            'rows': '{}'
        }

        # ================================================================
        # # Q17: getBetasByWordAndTopicId
        # ################################################################
        # # Get the topic-word distribution of a given word in a given topic
        # http://localhost:8983/solr/#/{model}/query?q=id:t{topic_id}&q.op=OR&indent=true&fl=payload(betas,{word})&useParams=
        # # Response example:
        # # # {
        # # #"responseHeader":{
        # # #    "zkConnected":true,
        # # #    "status":0,
        # # #    "QTime":3,
        # # #    "params":{
        # # #    "q":"id:t0",
        # # #    "indent":"true",
        # # #    "fl":"payload(betas, researchers)",
        # # #    "q.op":"OR",
        # # #    "useParams":"",
        # # #    "_":"1685958683375"}},
        # # #"response":{"numFound":1,"start":0,"numFoundExact":true,"docs":[
        # # #    {
        # # #        "payload(betas, researchers)":7.0}]
        # # #}}
        # ================================================================
        self.Q17 = {
            'q': 'id:t{}',
            'fl': 'payload(betas,{})',
        }
        
        
        #=================================================================
        # # Q19: getBOWbyDocsIDs  ##################################################################
        # # Get the bag of words of a list of documents (ids)
        # ================================================================
        self.Q18 = {
            'q': 'id:{}',
            'fl': 'payload(bow,{})',
        }

        #=================================================================
        # # Q20: getUserRelevantTopics  ##################################################################
        # # Get the topics that a user has marked as relevant
        # ================================================================
        self.Q19 = {
            'q': 'usersIsRelevant:{}',
            'fl': 'id,alphas,top_words_betas,topic_entropy,topic_coherence,ndocs_active,tpc_descriptions,tpc_labels,coords',
            'start': '{}',
            'rows': '{}'
        }
        
        
    def customize_Q1(self,
                     id: str,
                     model_name: str) -> dict:
        """Customizes query Q1 'getThetasDocById'.

        Parameters
        ----------
        id: str
            Document id.
        model_name: str
            Name of the topic model whose topic distribution is to be retrieved.

        Returns
        -------
        custom_q1: dict
            Customized query Q1.
        """

        custom_q1 = {
            'q': self.Q1['q'].format(id),
            'fl': self.Q1['fl'].format(model_name),
        }
        return custom_q1

    def customize_Q2(self,
                     corpus_name: str) -> dict:
        """Customizes query Q2 'getCorpusMetadataFields'

        Parameters
        ----------
        corpus_name: str
            Name of the corpus collection whose metadata fields are to be retrieved.

        Returns
        -------
        custom_q2: dict
            Customized query Q2.
        """

        custom_q2 = {
            'q': self.Q2['q'].format(corpus_name),
            'fl': self.Q2['fl'],
        }

        return custom_q2

    def customize_Q3(self) -> dict:
        """Customizes query Q3 'getNrDocsColl'

        Returns
        -------
        self.Q3: dict
            The query Q3 (no customization is needed).
        """

        return self.Q3

    def customize_Q4(self,
                     model_name: str,
                     topic: str,
                     threshold: str,
                     start: str,
                     rows: str) -> dict:
        """Customizes query Q4 'getDocsWithThetasLargerThanThr'

        Parameters
        ----------
        model_name: str
            Name of the topic model whose topic distribution is to be retrieved.
        topic: str
            Topic number.
        threshold: str
            Threshold value.
        start: str
            Start value.
        rows: str
            Number of rows to retrieve.

        Returns
        -------
        custom_q4: dict
            Customized query Q4.
        """

        custom_q4 = {
            'q': self.Q4['q'].format(model_name, str(threshold), str(topic)),
            'start': self.Q4['start'].format(start),
            'rows': self.Q4['rows'].format(rows),
        }
        return custom_q4

    def customize_Q5(self,
                     model_name: str,
                     thetas: str,
                     start: str,
                     rows: str) -> dict:
        """Customizes query Q5 'getDocsWithHighSimWithDocByid'

        Parameters
        ----------
        model_name: str
            Name of the topic model whose topic distribution is to be retrieved.
        thetas: str
            Topic distribution of the selected document.
        start: str
            Start value.
        rows: str
            Number of rows to retrieve.

        Returns
        -------
        custom_q5: dict
            Customized query Q5.
        """

        custom_q5 = {
            'q': self.Q5['q'].format(model_name, thetas),
            'fl': self.Q5['fl'].format(model_name),
            'start': self.Q5['start'].format(start),
            'rows': self.Q5['rows'].format(rows),
        }
        return custom_q5

    def customize_Q6(self,
                     id: str,
                     meta_fields: str) -> dict:
        """Customizes query Q6 'getMetadataDocById'


        Parameters
        ----------
        id: str
            Document id.
        meta_fields: str
            Metadata fields of the corpus collection to be retrieved.

        Returns
        -------
        custom_q6: dict
            Customized query Q6.
        """

        custom_q6 = {
            'q': self.Q6['q'].format(id),
            'fl': self.Q6['fl'].format(meta_fields)
        }

        return custom_q6

    def customize_Q7(self,
                     title_field: str,
                     string: str,
                     start: str,
                     rows: str) -> dict:
        """Customizes query Q7 'getDocsWithString'

        Parameters
        ----------
        title_field: str
            Title field of the corpus collection.
        string: str
            String to be searched in the title field.
        start: str
            Start value.
        rows: str
            Number of rows to retrieve.

        Returns
        -------
        custom_q7: dict
            Customized query Q7.
        """

        custom_q7 = {
            'q': self.Q7['q'].format(title_field, string),
            'fl': self.Q7['fl'],
            'start': self.Q7['start'].format(start),
            'rows': self.Q7['rows'].format(rows)
        }

        return custom_q7

    def customize_Q8(self,
                     start: str,
                     rows: str) -> dict:
        """Customizes query Q8 'getTopicsLabels'

        Parameters
        ----------
        rows: str
            Number of rows to retrieve.
        start: str
            Start value.

        Returns
        -------
        self.Q8: dict
            The query Q8
        """

        custom_q8 = {
            'q': self.Q8['q'],
            'fl': self.Q8['fl'],
            'start': self.Q8['start'].format(start),
            'rows': self.Q8['rows'].format(rows),
        }

        return custom_q8

    def customize_Q9(self,
                     model_name: str,
                     topic_id: str,
                     start: str,
                     rows: str) -> dict:
        """Customizes query Q9 'getDocsByTopic'

        Parameters
        ----------
        model_name: str
            Name of the topic model whose topic distribution is going to be used for retreving the top documents for the topic given by 'topic'.
        topic_id: str
            Topic number.
        start: str
            Start value.
        rows: str
            Number of rows to retrieve.

        Returns
        -------
        custom_q9: dict
            Customized query Q9.
        """

        custom_q9 = {
            'q': self.Q9['q'],
            'sort': self.Q9['sort'].format(model_name, topic_id),
            'fl': self.Q9['fl'].format(model_name, topic_id),
            'start': self.Q9['start'].format(start),
            'rows': self.Q9['rows'].format(rows),
        }
        
        return custom_q9

    def customize_Q10(self,
                      start: str,
                      rows: str,
                      only_id: bool) -> dict:
        """Customizes query Q10 'getModelInfo'

        Parameters
        ----------
        start: str
            Start value.
        rows: str

        Returns
        -------
        custom_q10: dict
            Customized query Q10.
        """

        if only_id:
            custom_q10 = {
                'q': self.Q10['q'],
                'fl': 'id',
                'start': self.Q10['start'].format(start),
                'rows': self.Q10['rows'].format(rows),
            }
        else:
            custom_q10 = {
                'q': self.Q10['q'],
                'fl': self.Q10['fl'],
                'start': self.Q10['start'].format(start),
                'rows': self.Q10['rows'].format(rows),
            }

        return custom_q10

    def customize_Q11(self,
                      topic_id: str) -> dict:
        """Customizes query Q11 'getBetasTopicById'.

        Parameters
        ----------
        topic_id: str
            Topic id.

        Returns
        -------
        custom_q11: dict
            Customized query Q11.
        """

        custom_q11 = {
            'q': self.Q11['q'].format(topic_id),
            'fl': self.Q11['fl']
        }
        return custom_q11

    def customize_Q12(self,
                      betas: str,
                      start: str,
                      rows: str) -> dict:
        """Customizes query Q12 'getMostCorrelatedTopics'

        Parameters
        ----------
        betas: str
            Word distribution of the selected topic.
        start: str
            Start value.
        rows: str
            Number of rows to retrieve.

        Returns
        -------
        custom_q11: dict
            Customized query q11.
        """

        custom_q12 = {
            'q': self.Q12['q'].format(betas),
            'fl': self.Q12['fl'],
            'start': self.Q12['start'].format(start),
            'rows': self.Q12['rows'].format(rows),
        }
        return custom_q12

    def customize_Q13(self,
                      model_name: str,
                      lower_limit: str,
                      upper_limit: str,
                      year: str,
                      start: str,
                      rows: str) -> dict:
        
        """Customizes query Q13 'getPairsOfDocsWithHighSim'

        Parameters
        ----------
        model_name: str
            Name of the topic model where semantic similarity is evaluated.
        lower_limit: str
            Lower percentage of semantic similarity to return pairs of documents.
        upper_limit: str
            Upper percentage of semantic similarity to return pairs of documents.
        year: str
            Year to filter documents.
        start: str
            Start value.
        rows: str
            Number of rows to retrieve.

        Returns
        -------
        custom_q13: dict
            Customized query Q13.
        """

        if year:
            custom_q13 = {
                'q': self.Q13['q'].format(model_name, lower_limit, upper_limit, year, year),
                'fl': self.Q13['fl'].format(model_name),
                'start': self.Q13['start'].format(start),
                'rows': self.Q13['rows'].format(rows),
            }
        else:
            custom_q13 = {
                'q': self.Q13['q_no_date'].format(model_name, lower_limit, upper_limit),
                'fl': self.Q13['fl'].format(model_name),
                'start': self.Q13['start'].format(start),
                'rows': self.Q13['rows'].format(rows),
            }
        
        return custom_q13

    def customize_Q14(self,
                      model_name: str,
                      thetas: str,
                      start: str,
                      rows: str) -> dict:
        """Customizes query Q14 'getDocsSimilarToFreeText'

        Parameters
        ----------
        model_name: str
            Name of the topic model whose topic distribution is to be retrieved.
        thetas: str
            Topic distribution of the user's free text.
        start: str
            Start value.
        rows: str
            Number of rows to retrieve.

        Returns
        -------
        custom_q14: dict
            Customized query Q14.
        """

        custom_q14 = {
            'q': self.Q14['q'].format(model_name, thetas),
            'fl': self.Q14['fl'].format(model_name),
            'start': self.Q14['start'].format(start),
            'rows': self.Q14['rows'].format(rows),
        }
        return custom_q14

    def customize_Q15(self,
                      id: str) -> dict:
        """Customizes query Q15 'getLemmasDocById'.

        Parameters
        ----------
        id: str
            Document id.

        Returns
        -------
        custom_q15: dict
            Customized query Q15.
        """

        custom_q15 = {
            'q': self.Q15['q'].format(id),
            'fl': self.Q15['fl'],
        }
        return custom_q15

    def customize_Q16(self,
                      model_name: str,
                      start: str,
                      rows: str) -> dict:
        """Customizes query Q16 'getThetasAndDateAllDocs'.

        Parameters
        ----------
        model_name: str
            Name of the topic model whose topic distribution is to be retrieved.
        start: str
            Start value.
        rows: str
            Number of rows to retrieve.

        Returns
        -------
        custom_q16: dict
            Customized query Q1.
        """

        custom_q16 = {
            'q': self.Q16['q'],
            'fl': self.Q16['fl'].format(model_name),
            'start': self.Q16['start'].format(start),
            'rows': self.Q16['rows'].format(rows),
        }
        return custom_q16

    def customize_Q17(self,
                      topic_id: str,
                      word: str) -> dict:
        """Customizes query Q17 'getBetasByWordAndTopicId'.

        Parameters
        ----------
        topic_id: str
            Topic id.
        word: str
            Word.

        Returns
        -------
        custom_q17: dict
            Customized query Q17.
        """

        custom_q17 = {
            'q': self.Q17['q'].format(topic_id),
            'fl': self.Q17['fl'].format(word)
        }

        return custom_q17
    
    def customize_Q18(self,
                      ids: str,
                      words: str,
                      start:str,
                      rows: str) -> dict:
    
        
        custom_q18 = {
            'q':  self.Q18['q'].format(' & id:'.join(ids)),
            'fl': 'id, ' + ', '.join(self.Q18['fl'].format(word) for word in words),
            'start': self.Q16['start'].format(start),
            'rows': self.Q16['rows'].format(rows),
        }

        return custom_q18
    

    def customize_Q19(self,
                      start: str,
                      rows: str,
                      user: str) -> dict:
        """Customizes query Q19

        Parameters
        ----------
        start: str
            Start value.
        rows: str
            Number of rows to retrieve.
        user: str
            User name

        Returns
        -------
        custom_q19: dict
            Customized query Q19.
        """

        custom_q19 = {
            'q': self.Q19['q'].format(user),
            'fl': self.Q19['fl'],
            'start': self.Q19['start'].format(start),
            'rows': self.Q19['rows'].format(rows),
        }

        return custom_q19