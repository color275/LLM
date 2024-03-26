import streamlit as st
import fitz
import logging
import boto3
import json
from datetime import datetime
from langchain_community.chat_models import BedrockChat
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms.bedrock import Bedrock
from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.helpers import bulk

opensearch_username = 'admin'
opensearch_password = 'Admin12#$'
# opensearch_endpoint = 'vpc-xxxx-xxxxxxx.ap-northeast-2.es.amazonaws.com'
opensearch_endpoint = 'search-chiholee-7oq7vwq7wjmoxx3uunmpacvjfa.ap-northeast-2.es.amazonaws.com'
index_name = 'llm'
bedrock_region = 'us-west-2'
stop_record_count = 100
record_stop_yn = False
bedrock_model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
bedrock_embedding_model_id = "amazon.titan-embed-text-v1"

opensearch_client = OpenSearch(
    hosts=[{
        'host': opensearch_endpoint,
        'port': 443
    }],
    http_auth=(opensearch_username, opensearch_password),
    index_name=index_name,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
    timeout=30
)

index_name = 'python-test-index'
index_body = {
    'settings': {
        'index': {
            'number_of_shards': 2
        }
    }
}

# response = opensearch_client.indices.create(index_name, body=index_body)
# print('\nCreating index:')
# print(response)

opensearch_client.create_index_mapping(opensearch_client, index_name)
