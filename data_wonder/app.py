# export AWS_DEFAULT_REGION='us-west-2'
# nohup streamlit run app.py --server.port 8503 &
# ssh -i /Users/chiholee/Desktop/Project/keys/key.pem -L 13306:ecommerce.cluster-cgkgybnzurln.ap-northeast-2.rds.amazonaws.com:3306 ec2-user@3.35.137.48

import streamlit as st
import fitz
import logging
import boto3
import json
import os
import re
import pymysql
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
from langchain_community.chat_models import BedrockChat
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms.bedrock import Bedrock
from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.helpers import bulk
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

INIT_MESSAGE = {"role": "assistant",
                "type": "text",
                "content": """
안녕하세요. 저는 <font color='red'><b>Amazon Bedrock과 Claude3</b></font>를 활용해서 여러분들이 찾고 싶은 데이터를 대신 찾아줄 <i><b>[데이터가 궁금해]<i><b> 입니다. 
<br>아래와 같이 질문해보세요.
- <font color='#32CD32;'><b>어제 판매된 상품 기준으로 주문 금액 TOP 5 를 알려줘</b></font><br>
- <font color='#32CD32;'><b>지난 일주일간 주문 실적을 일 별로 알려줘</b></font><br>
- <font color='#32CD32;'><b>최근 5분 동안 총주문금액과 총주문수량을 분 단위로 알려줘</b></font><br>
- <font color='#32CD32;'><b>오늘 총 주문금액이 가장 적은 상품을 알려줘</b></font><br>
---
무엇을 도와드릴까요?"""}



################################################################################

load_dotenv()
opensearch_username = os.getenv('OPENSEARCH_USERNAME')
opensearch_password = os.getenv('OPENSEARCH_PASSWORD')
opensearch_endpoint = os.getenv('OPENSEARCH_ENDPOINT')
index_name = os.getenv('OPENSEARCH_INDEX_NAME')
mysql_host = os.getenv('MYSQL_HOST')
mysql_port = os.getenv('MYSQL_PORT')
mysql_user = os.getenv('MYSQL_USER')
mysql_password = os.getenv('MYSQL_PASSWORD')
mysql_db = os.getenv('MYSQL_DB')

bedrock_region = 'us-west-2'
stop_record_count = 100
record_stop_yn = False
bedrock_model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
bedrock_embedding_model_id = "amazon.titan-embed-text-v1"
################################################################################


def get_opensearch_cluster_client():
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
    return opensearch_client


def get_bedrock_client():
    bedrock_client = boto3.client(
        "bedrock-runtime", region_name=bedrock_region)
    return bedrock_client


def create_langchain_vector_embedding_using_bedrock(bedrock_client):
    bedrock_embeddings_client = BedrockEmbeddings(
        client=bedrock_client,
        model_id=bedrock_embedding_model_id)
    return bedrock_embeddings_client


def create_opensearch_vector_search_client(bedrock_embeddings_client, _is_aoss=False):
    docsearch = OpenSearchVectorSearch(
        index_name=index_name,
        embedding_function=bedrock_embeddings_client,
        opensearch_url=f"https://{opensearch_endpoint}",
        http_auth=(opensearch_username, opensearch_password),
        is_aoss=_is_aoss
    )
    return docsearch


def create_bedrock_llm():
    # claude-2 이하
    # bedrock_llm = Bedrock(
    #     model_id=model_version_id,
    #     client=bedrock_client,
    #     model_kwargs={'temperature': 0}
    #     )
    # bedrock_llm = BedrockChat(model_id=model_version_id, model_kwargs={'temperature': 0}, streaming=True)

    bedrock_llm = BedrockChat(
        model_id=bedrock_model_id, 
        model_kwargs={'temperature': 0},
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
        )
    return bedrock_llm


def get_bedrock_client():
    bedrock_client = boto3.client(
        "bedrock-runtime", region_name=bedrock_region)
    return bedrock_client


def create_vector_embedding_with_bedrock(text, bedrock_client):
    payload = {"inputText": f"{text}"}
    body = json.dumps(payload)
    modelId = "amazon.titan-embed-text-v1"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_client.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    embedding = response_body.get("embedding")
    return {"_index": index_name, "text": text, "vector_field": embedding}


def extract_sentences_from_pdf(opensearch_client, pdf_file, progress_bar, progress_text):
    try:
        logging.info(
            f"Checking if index {index_name} exists in OpenSearch cluster")

        exists = opensearch_client.indices.exists(index=index_name)

        if not exists:
            body = {
                'settings': {
                    'index': {
                        'number_of_shards': 3,
                        'number_of_replicas': 2,
                        "knn": True,
                        "knn.space_type": "cosinesimil"
                    }
                }
            }
            success = opensearch_client.indices.create(index_name, body=body)
            if success:
                body = {
                    "properties": {
                        "vector_field": {
                            "type": "knn_vector",
                            "dimension": 1536
                        },
                        "text": {
                            "type": "keyword"
                        }
                    }
                }
                success = opensearch_client.indices.put_mapping(
                    index=index_name,
                    body=body
                )

        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        all_records = []
        for page in doc:
            all_records.append(page.get_text())

        # URL Scraping
        # doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        # all_text = ""
        # for page in doc:
        #     all_text += page.get_text()
        # doc.close()
        # all_records = re.split(r'(?<=[.!?])\s+', all_text)

        logging.info(f"PDF LIST 개수 : {len(all_records)}")

        total_records = len(all_records)
        processed_records = 0

        bedrock_client = get_bedrock_client()

        all_json_records = []

        for record in all_records:
            if record_stop_yn and processed_records > stop_record_count:

                success, failed = bulk(opensearch_client, all_json_records)
                break

            records_with_embedding = create_vector_embedding_with_bedrock(
                record, bedrock_client)
            all_json_records.append(records_with_embedding)

            processed_records += 1
            progress = int((processed_records / total_records) * 100)
            progress_bar.progress(progress)

            if processed_records % 500 == 0 or processed_records == len(all_records):

                success, failed = bulk(opensearch_client, all_json_records)
                all_json_records = []

        progress_text.text("완료")
        logging.info("임베딩을 사용하여 레코드 생성 완료")

        return total_records
    except Exception as e:
        print(str(e))
        st.error('PDF를 임베딩 하는 과정에서 오류가 발생되었습니다.')
        return 0


def find_answer_in_sentences(question):
    try:
        # question = question + " 정보가 없다는 이야기는 하지 말고, 제공된 정보를 바탕으로 너가 SQL을 만들어줘."
        question = question
        bedrock_client = get_bedrock_client()
        bedrock_llm = create_bedrock_llm()

        bedrock_embeddings_client = create_langchain_vector_embedding_using_bedrock(
            bedrock_client)

        opensearch_vector_search_client = create_opensearch_vector_search_client(
            bedrock_embeddings_client)
        
        
        # Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. don't include harmful content        
        # SQL 생성 시 최대 10개의 레코드만 표시되도록 SQL의 가장 하단에 Limit 10을 SQL에 포함시켜줘. 그리고 Limit 10을 임의로 추가했다고 초록색 글자로 안내해줘.
        prompt_template = """
        Use the following pieces of context to answer the question at the end.
        인사를 하면 별도 데이터나 SQL 제공없이 간단한 소개만 해줘.
        질문에 대해 설명을 앞부분에 작성해줘.
        데이터를 알려달라고 할 때만 질문을 확인할 수 있는 SQL을 만들어서 제공해줘.
        SQL은 제공한 테이블만 사용해서 만들어줘
        SQL은 markdown의 코드 sql 태그 안에 넣어줘.
        markdown 코드 sql 태그 안에는 하나의 sql 만 넣어줘.
        SELECT 절에서 컬럼 AS 뒤의 alias는 한글로 작성해줘.
        where 절에서 컬럼을 형변환을 위해 가공하지 말아줘. 예를 들어 order_dt 를 DATE(order_dt)로 변환하지 말아라.
        where 컬럼A = 상수B 일 떄 컬럼A와 상수B의 데이터타입이 다르다면 상수B를 형변환해서 데이터타입을 동일하게 맞춰줘.예를 들어 order_dt = CURDATE() 일 경우 order_dt = date_format(curdate(), '%Y-%m-%d') 로 사용해줘
        where 에서 동일한 데이터타입으로 비교를 할 수 있게 적절한 형변환을 해줘.
        SQL의 FROM 절에서 테이블의 alias 는 반드시 적어줘.
        SQL 생성 시 별도의 정렬이 필요없다면 [alias].last_update_time desc 을 정렬 조건으로 추가해줘. [alias] 에는 적절한 테이블의 alias를 명시해줘.
        {context}

        Question: {question}
        Answer:"""
        
        prompt_template = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        # prompt = prompt_template.format(
        #     context=CONTEXT_DATA, question=question)
        
        # print("# prompt : ", prompt)


        logging.info(
            f"Starting the chain with KNN similarity using OpenSearch, Bedrock FM {bedrock_model_id}, and Bedrock embeddings with {bedrock_embedding_model_id}")

        qa = RetrievalQA.from_chain_type(llm=bedrock_llm,
                                         chain_type="stuff",
                                         retriever=opensearch_vector_search_client.as_retriever(),
                                         return_source_documents=True,
                                         chain_type_kwargs={
                                             "prompt": prompt_template, "verbose": True},
                                         verbose=True)

        # response = qa({"context": CONTEXT_DATA, "query": question},
        #               return_only_outputs=False)
        response = qa(question,
                      return_only_outputs=False)

        source_documents = response.get('source_documents')
        # logging.info(f"The answer from Bedrock {bedrock_model_id} is: {response.get('result')}")
        return f"{response.get('result')}"
    except Exception as e:
        if 'index_not_found_exception' in str(e):
            st.error('인덱스를 찾을 수 없습니다. PDF 파일을 업로드 했는지 확인해주세요')
        else:
            print(str(e))
            st.error('답변을 찾는 과정에서 예상치 못한 오류가 발생했습니다.')
        return "오류로 인해 답변을 제공할 수 없습니다."


# def execute_query(sql):
#     try:
#         # 데이터베이스 연결
#         connection = pymysql.connect(host=mysql_host,
#                                      port=int(mysql_port),
#                                      user=mysql_user,
#                                      password=mysql_password,
#                                      database=mysql_db,
#                                      cursorclass=pymysql.cursors.DictCursor)

#         with connection:
#             with connection.cursor() as cursor:
#                 # SQL 쿼리 실행
#                 cursor.execute(sql)
#                 result = cursor.fetchall()  # 모든 결과를 가져옴

#                 # 결과를 Streamlit에 표시
#                 if result:
#                     st.write(result)
#                 else:
#                     st.write("쿼리 결과가 없습니다.")

#     except Exception as e:
#         st.error(f"데이터베이스 쿼리 실행 중 오류 발생: {e}")


def connect_to_database():
    return pymysql.connect(
        host=mysql_host,
        port=int(mysql_port),
        user=mysql_user,
        password=mysql_password,
        database=mysql_db,
        charset='utf8mb4'
    )

# SQL 쿼리 실행 및 결과를 데이터프레임으로 변환


def execute_query_and_return_df(sql):
    conn = connect_to_database()
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql)
            result = cursor.fetchall()
            df = pd.DataFrame(result, columns=[i[0]
                              for i in cursor.description])
    finally:
        conn.close()
    return df


def main():

    # 기존 업로드 문서 삭제
    # if st.sidebar.button("기존 업로드 문서 삭제"):
    #     response = opensearch_client.delete_opensearch_index(opensearch_client, index_name)
    #     # st.session_state['question'] = ""  # 질문 세션 상태 초기화
    #     if response:
    #         logging.info("OpenSearch index successfully deleted")
    #         st.sidebar.success("OpenSearch 인덱스가 성공적으로 삭제되었습니다.")  # 성공 알림 추가

    opensearch_client = get_opensearch_cluster_client()
    st.set_page_config(page_title='🤖 Chat with Bedrock', layout='wide')
    # st.header('_Chatbot_ using :blue[OpenSearch] :sunglasses:', divider='rainbow')
    st.header(':blue[데이터가] _궁금해_ :sunglasses:', divider='rainbow')    

    if 'qa_history' not in st.session_state:
        st.session_state.qa_history = []

    with st.sidebar:
        st.sidebar.markdown(
            ':smile: **Createby:** chiholee@amazon.com', unsafe_allow_html=True)
        st.sidebar.markdown('---')
        st.title("RAG Embedding")
        pdf_file = st.file_uploader(
            "PDF 업로드를 통해 추가 학습을 할 수 있습니다.", type=["pdf"], key=None)
        

        if 'last_uploaded' not in st.session_state:
            st.session_state.last_uploaded = None

        if pdf_file is not None and pdf_file != st.session_state.last_uploaded:
            progress_text = st.empty()
            st.session_state['progress_bar'] = st.progress(0)
            progress_text.text("RAG(OpenSearch) 임베딩 중...")
            record_cnt = extract_sentences_from_pdf(
                opensearch_client, pdf_file, st.session_state['progress_bar'], progress_text)
            if record_cnt > 0:
                st.session_state['processed'] = True
                st.session_state['record_cnt'] = record_cnt
                st.session_state['progress_bar'].progress(100)
                st.session_state.last_uploaded = pdf_file
                st.success(f"{record_cnt} Vector 임베딩 완료!")

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [INIT_MESSAGE]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["type"] == "text":
                st.markdown(message["content"], unsafe_allow_html=True)
            elif message["type"] == "data":
                st.markdown(message["content"][0], unsafe_allow_html=True)
                st.write(':bulb: **쿼리 실행 결과**', message["content"][1])

    question = st.chat_input("Say something")

    if question:
        st.session_state.messages.append({"role": "user",
                                          "type": "text",
                                          "content": question})
        with st.chat_message("user"):
            st.markdown(question, unsafe_allow_html=True)

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = find_answer_in_sentences(question)

                try :
                    sql_queries = re.findall(r'```sql(.*?)```', answer, re.DOTALL)
                    st.markdown(answer, unsafe_allow_html=True)

                    if len(sql_queries) > 0:
                        # st.markdown(sql_queries[0], unsafe_allow_html=True)
                        sql = sql_queries[0]
                        df = execute_query_and_return_df(sql)
                        st.write(':bulb: **쿼리 실행 결과**', df)

                        message = {"role": "assistant",
                                "type": "data",
                                "content": [answer, df]}

                    else:
                        message = {"role": "assistant",
                                "type": "text",
                                "content": answer}

                    st.session_state.messages.append(message)
                except Exception as e:
                    pass


if __name__ == "__main__":
    main()
