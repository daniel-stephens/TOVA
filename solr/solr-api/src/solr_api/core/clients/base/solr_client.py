"""
This  module provides 3 generic classes to handle Solr API responses and requests.

The SolrResults class is for wrapping decoded Solr responses, where individual documents can be retrieved either through the docs attribute or by iterating over the instance. 

The SolrResp class is for handling Solr API response and errors. 

The SolrClient class is for handling Solr API requests. 

Author: Lorena Calvo-Bartolomé
Date: 27/03/2023
Modifed: 24/01/2024 (Updated for NP-Solr-Service (NextProcurement Proyect))
"""

import json
import logging
import os
from typing import List, Union
from urllib import parse
from typing import List
from urllib.parse import quote

import requests


class SolrResults(object):
    """Class for wrapping decoded (from JSON) solr responses.

    Individual documents can be retrieved either through ``docs`` attribute
    or by iterating over results instance.
    """

    def __init__(self,
                 json_response: dict,
                 next_page_query: bool = None) -> None:
        """Init method.

        Parameters
        ----------
        json_response: dict
            JSON response from Solr.
        next_page_query: bool, defaults to None
            If True, then the next page of results is fetched.
        """
        self.solr_json_response = json_response

        # Main response part of decoded Solr response
        response = json_response.get("response") or {}
        self.docs = response.get("docs", ())
        self.hits = response.get("numFound", 0)

        # other response metadata
        self.debug = json_response.get("debug", {})
        self.highlighting = json_response.get("highlighting", {})
        self.facets = json_response.get("facet_counts", {})
        self.spellcheck = json_response.get("spellcheck", {})
        self.stats = json_response.get("stats", {})
        self.qtime = json_response.get("responseHeader", {}).get("QTime", None)
        self.grouped = json_response.get("grouped", {})
        self.nextCursorMark = json_response.get("nextCursorMark", None)
        self._next_page_query = (
            self.nextCursorMark is not None and next_page_query or None
        )

        return

    def __len__(self) -> int:
        """Return the number of documents in the results."""
        if self._next_page_query:
            return self.hits
        else:
            return len(self.docs)

    def __iter__(self) -> iter:
        """Iterate over the documents in the results."""
        result = self
        while result:
            for d in result.docs:
                yield d
            result = result._next_page_query and result._next_page_query()


class SolrResp(object):
    """
    A class to handle Solr API response and errors.

    Examples
    --------
        # From delete collection
        response = {
                    "responseHeader":{
                        "status":0,
                        "QTime":1130}
                    }
        # From query
        response = {
                    "responseHeader":{
                        "zkConnected":true,
                        "status":0,
                        "QTime":15,
                        "params":{
                        "q":"*:*",
                        "indent":"true",
                        "q.op":"OR",
                        "useParams":"",
                        "_":"1681211825934"
                    }},
                        "response":{
                            "numFound":3,
                            "start":0,
                            "numFoundExact":true,
                            'docs': [{'id': 1}, {'id': 2}, {'id': 3}]
                      }}
    """

    def __init__(self,
                 status_code: int,
                 text: str,
                 data: list,
                 results: SolrResults = None) -> None:
        """Init method.

        Parameters
        ----------
        status_code: int
            The status code of the Solr API response.
        text: str
            The text of the Solr API response.
        data: list
            A list of dictionaries that represents the data returned by the Solr API response (e.g., when list_collections is used)
        results: SolrResults
            A SolrResults object that represents the data returned by the Solr API response, only under the condition that "response" is in the JSON dict returned by Solr (e.g., when performing a query)
        """
        self.status_code = status_code
        self.text = text
        self.data = data
        self.results = results

        return

    @staticmethod
    def from_error(status_code: int, text: str):
        return SolrResp(status_code, text, [])

    @staticmethod
    def from_requests_response(resp: requests.Response, logger: logging.Logger):
        """
        Parameters
        ----------
        resp : requests.Response
            The Solr API response.
        logger : logging.Logger
            The logger object to log messages and errors.
        """

        status_code = 400
        data = []
        text = ""
        results = {}

        # Get JSON object of the result
        resp = resp.json()

        # If response header has status 0, request is acknowledged
        if 'responseHeader' in resp and resp['responseHeader']['status'] == 0:
            logger.info('-- -- Request acknowledged')
            status_code = 200
        else:
            # If there is an error in response header, set status code and text attributes accordingly
            status_code = resp['responseHeader']['status']
            text = resp['error']['msg']
            logger.error(
                f'-- -- Request generated an error {status_code}: {text}')

        # If collections are returned in response, set data attribute to collections list
        if 'collections' in resp:
            data = resp['collections']

        if 'response' in resp:
            results = SolrResults(resp, True)

        return SolrResp(status_code, text, data, results)

    def raise_for_status(self):
        """Raise an HTTPError if the response indicates failure."""
        if self.status_code is None:
            raise requests.HTTPError("No status_code on SolrResp")
        if int(self.status_code) >= 400:
            # include body text if available
            msg = f"Solr error {self.status_code}"
            if self.text:
                msg += f": {self.text}"
            raise requests.HTTPError(msg)

class SolrClient(object):
    """
    A class to handle Solr API requests.
    """

    def __init__(self, logger: logging.Logger) -> None:
        """
        Parameters
        ----------
        logger : logging.Logger
            The logger object to log messages and errors.
        """

        # Get the Solr URL from the environment variables
        self.solr_url = os.environ.get('SOLR_URL')

        # Initialize requests session and logger
        self.solr = requests.Session()
        # self.logger = logger
        import logging
        logging.basicConfig(level='DEBUG')
        self.logger = logging.getLogger('Solr')

        return

    def _do_request(self,
                    type: str,
                    url: str,
                    timeout: int = None,
                    **params) -> SolrResp:
        """Sends a requests to the given url with the given params and returns an object of the SolrResp class

        Parameters
        ----------
        type: str
            The type of request to send.
        url: str
            The url to send the request to.
        timeout: int, defaults to 10
            The timeout in seconds to use for the request.

        Returns
        -------
        SolrResp : SolrResp
            The response object.
        """

        # Send request
        if type == "get":
            resp = requests.get(
                url=url,
                timeout=timeout,
                **params
            )
            pass
        elif type == "post":
            resp = requests.post(
                url=url,
                timeout=timeout,
                **params
            )
        elif type == "delete":
            resp = requests.delete(
                url=url,
                timeout=timeout,
                **params
            )
        else:
            self.logger.error(f"-- -- Invalid type {type}")
            return

        # Parse Solr response
        solr_resp = SolrResp.from_requests_response(resp, self.logger)

        return solr_resp

    # ======================================================
    # MANAGING (Creation, deletion, listing, etc.)
    # ======================================================
    def add_field_to_schema(self,
                            col_name: str,
                            field_name: str,
                            field_type: str) -> Union[List[dict], int]:
        """Adds a field of type 'field_type'  and name 'field_name' to the schema of the collection given by 'col_name'. 

        Parameters
        ----------
        col_name: str
            The name of the collection to add the field to.
        field_name: str
            The name of the field to add.
        field_type: str
            The type of the field to add.

        Returns
        -------
        List[dict]
            A list of dictionaries that represents the data returned by the Solr API response.
        int
            The HTTP status code of the Solr API response.
        """

        headers_ = {"Content-Type": "application/json"}
        data = {
            "add-field": {
                "name": field_name,
                "type": field_type,
                "indexed": "true",
                "termOffsets": "true",
                "stored": "true",
                "termPositions": "true",
                "termVectors": "true",
                "multiValued": "false"
            }
        }
        url_ = '{}/api/collections/{}/schema?'.format(self.solr_url, col_name)

        # Send request to Solr
        solr_resp = self._do_request(type="post", url=url_,
                                     headers=headers_, json=data)

        return [{'name': col_name}], solr_resp.status_code

    def delete_field_from_schema(self,
                                 col_name: str,
                                 field_name: str) -> Union[List[dict], int]:
        """Deletes a field of name 'field_name' from the schema of the collection given by 'col_name'. 

        Parameters
        ----------
        col_name: str
            The name of the collection to delete the field from.
        field_name: str
            The name of the field to delete.

        Returns
        -------
        List[dict]
            A list of dictionaries that represents the data returned by the Solr API response.
        int
            The HTTP status code of the Solr API response.
        """

        headers_ = {"Content-Type": "application/json"}
        data = {
            "delete-field": {
                "name": field_name,
            }
        }
        url_ = '{}/api/collections/{}/schema?'.format(self.solr_url, col_name)

        # Send request to Solr
        solr_resp = self._do_request(type="post", url=url_,
                                     headers=headers_, json=data)

        return [{'name': col_name}], solr_resp.status_code

    def create_collection(self,
                          col_name: str,
                          config: str = '_default',
                          nshards: int = 1,
                          replicationFactor: int = 1) -> Union[str, int]:
        """Creates a Solr collection with the given name, config, number of shards, and replication factor.
        Returns a list with a dictionary containing the name of the created collection and the HTTP status code.

        Parameters
        ----------
        col_name: str
            The name of the collection to create.
        config: str, defaults to '_default'
            The name of the config to use for the collection.
        nshards: int, defaults to 1
            The number of shards to use for the collection.
        replicationFactor: int, defaults to 1
            The replication factor to use for the collection.

        Returns
        -------
        str
            The name of the created collection.
        int
            The HTTP status code of the Solr API response.
        """

        # Check if collection already exists
        colls, _ = self.list_collections()
        if col_name in colls:
            solr_resp = SolrResp.from_error(
                409, "Collection {} already exists".format(col_name))
            return _, solr_resp.status_code
    
        # Carry on with creation if collection does not exists
        headers_ = {"Content-Type": "application/json"}
        data = {
            "name": col_name,
            "config": config,             
            "numShards": nshards,
            "replicationFactor": replicationFactor
        }
        url_ = f"{self.solr_url}/api/collections"

        solr_resp = self._do_request(type="post", url=url_,
                                    headers=headers_, json=data)
        return col_name, solr_resp.status_code


    def delete_collection(self, col_name: str) -> Union[List[dict], int]:
        """
        Deletes a Solr collection with the given name.
        Returns a list with a dictionary containing the name of the deleted collection and the HTTP status code.

        Parameters
        ----------
        col_name: str
            The name of the collection to delete.

        Returns
        -------
        List[dict]
            A list of dictionaries with the name of the deleted collection.
        """

        url_ = f"{self.solr_url}/api/collections/{quote(col_name)}"
        solr_resp = self._do_request(type="delete", url=url_)

        return col_name, solr_resp.status_code

    def delete_doc_by_id(self, col_name: str, id: int) -> int:
        """
        Deletes the document with the given id in the Solr collection with the given name. 

        Parameters
        ----------
        col_name: str
            The name of the collection to delete the document from.
        id: int
            The id of the document to delete.

        Returns
        -------
        int 
            The HTTP status code of the Solr API response.
        """

        headers_ = {"Content-Type": "application/xml"}
        data_ = "<delete><query>(id:" + id + ")</query></delete>"
        params_ = {
            'commitWithin': '1000',
            'overwrite': 'true',
            'wt': 'json'
        }

        url_ = '{}/solr/{}/update'.format(self.solr_url, col_name)

        # Send request to Solr
        solr_resp = self._do_request(type="post", url=url_,
                                     headers=headers_, data=data_, params=params_)

        return solr_resp.status_code

    def list_collections(self) -> Union[List[dict], int]:
        """
        Lists all Solr collections and returns a list of dictionaries, where each dictionary has a key "name" with the value of the collection name,
        and the HTTP status code.

        Returns
        -------
        List[dict]
            A list of dictionaries with the names of the collections.
        """

        url_ = '{}/api/collections'.format(self.solr_url)

        # Send request to Solr
        solr_resp = self._do_request(type="get", url=url_)

        return solr_resp.data, solr_resp.status_code

    # ======================================================
    # INDEXING
    # ======================================================

    def index_batch(self,
                    docs_batch: List[dict],
                    col_name: str,
                    to_index: int,
                    index_from: int,
                    index_to: int) -> int:
        """Takes a batch of documents, a Solr collection name, and the indices of the batch to be indexed, and sends a POST request to the Solr server to index the documents. The method returns the status code of the response.

        Parameters
        ----------
        docs_batch : list[dict])
            A list of dictionaries where each dictionary represents a document to be indexed.
        col_name : str
            The name of the Solr collection to index the documents into.
        to_index : int
            The total number of documents to be indexed.
        index_from :int
            The starting index of the documents in the batch to be indexed.
        index_to: int
            The ending index of the documents in the batch to be indexed.

        Returns
        -------
        sc : int
            The status code of the response.
        """

        headers_ = {'Content-type': 'application/json'}

        params = {
            'commitWithin': '1000',
            'overwrite': 'true',
            'wt': 'json'
        }

        url_ = '{}/solr/{}/update'.format(self.solr_url, col_name)

        # Send request to Solr
        solr_resp = self._do_request(
            type="post", url=url_, headers=headers_, 
            json=docs_batch,
            params=params, proxies={})

        if solr_resp.status_code == 200:
            self.logger.info(
                f"-- -- Indexed documents from {index_from} to {index_to} / {to_index} in Collection '{col_name}'")

        return solr_resp.status_code

    def index_documents(self,
                        json_docs: List[dict],
                        col_name: str,
                        batch_size: int = 100) -> None:
        """It takes a list of documents in JSON format and a Solr collection name, splits the list into batches, and sends a POST request to the Solr server to index the documents in batches. The method returns the status code of the response.

        Parameters
        ----------
        json_docs : list[dict]
            A list of dictionaries where each dictionary represents a document to be indexed.
        col_name : str 
            The name of the Solr collection to index the documents into.
        batch_size : int
            Batch size with which the documents will be indexed
        """

        docs_batch = []
        index_from = 0
        to_index = len(json_docs)
        for index, doc in enumerate(json_docs):

            docs_batch.append(doc)

            if index % batch_size == 0 and index != 0:
                # Index batch to Solr
                self.index_batch(docs_batch, col_name, to_index,index_from=index_from, index_to=index)
                docs_batch = []
                index_from = index + 1
                self.logger.info("==== indexed {} documents ======" .format(index))
        # To index the rest, when 'documents' list < batch_size.
        self.logger.info("Docs batch: {}".format(len(docs_batch)))
        if docs_batch:
            self.index_batch(docs_batch, col_name, to_index,index_from=index_from, index_to=index)
        self.logger.info("-- -- Finished indexing")

        return

    # ======================================================
    # QUERIES
    # ======================================================
    def execute_query(
            self,
            q: str,
            col_name: str,
            type="get",
            json_body: dict = None,
            **kwargs) -> Union[int, SolrResults]:
        """ 
        Performs a query and returns the results.

        Requires a ``q`` for a string version of the query to run. Optionally accepts ``**kwargs``for additional options to be passed through the Solr URL.

        Parameters
        ----------
        q : str
            The query to be executed.
        col_name : str
            The name of the Solr collection to query.
        json_body : dict, optional
            A dictionary representing the JSON body to be sent in a POST request.
        **kwargs
            Additional options to be passed through the Solr URL.

        Returns
        -------
        int
            The HTTP status code of the Solr API response.
        SolrResults
            The results of the query.

        Usage
        -----
            # All docs
            results = solr.execute_query('*:*')
        """

        # Prepare query
        if type.lower() == "get":
            params = {"q": q}
            params.update(kwargs)

            # We want the result of the query as json
            params["wt"] = "json"

            # Encode query for GET request
            self.logger.info(params)
            query_string = parse.urlencode(params)

        else:
            params = q
            query_string = parse.urlencode(params)

        self.logger.info(query_string)

        # Construct the URL
        url_ = '{}/solr/{}/select?{}'.format(self.solr_url,
                                             col_name, query_string)

        # Set headers for JSON content
        headers = {"Content-Type": "application/json"}

        if type.lower() == "post" and json_body:
            # Send query to Solr as a POST request with a JSON body
            solr_resp = self._do_request(
                type=type, url=url_, headers=headers, data=json.dumps(json_body))
        else:
            # Send query to Solr as a GET request or POST without a body
            solr_resp = self._do_request(type=type, url=url_, headers=headers)

        return solr_resp.status_code, solr_resp.results