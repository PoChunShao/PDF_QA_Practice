import neo4j
import gradio as gr
import psycopg2
import time
import re
import pandas as pd
import voyageai
import os

from elasticsearch import Elasticsearch
from openai import OpenAI
from dotenv import load_dotenv
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_query import get_recommendations_v2

load_dotenv('.env', override=True)

# voyage rerank config
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
vo = voyageai.Client(api_key=VOYAGE_API_KEY)

# openai config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# neo4j config
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD") 
driver = neo4j.GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
embedder = OpenAIEmbeddings(model="text-embedding-3-large")

# elaticsearch config
ES_HOST = os.getenv("ES_HOST")
es_client = Elasticsearch(
    hosts=ES_HOST
)

# postgresql config
PG_HOST = os.getenv("PG_HOST")
PG_DATABASE = os.getenv("PG_DATABASE")
PG_USERNAME = os.getenv("PG_USERNAME")
PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_PORT = os.getenv("PG_PORT")
PG_CHUNK_TABLE_NAME = os.getenv("PG_CHUNK_TABLE_NAME")
PG_FORM_TABLE_NAME = os.getenv("PG_FORM_TABLE_NAME")
connection = psycopg2.connect(
    dbname=PG_DATABASE,
    host=PG_HOST,
    port=PG_PORT,
    user=PG_USERNAME,
    password=PG_PASSWORD
)

cur = connection.cursor()

def hybrid_search(message,history):
    pattern = r"^(?![0-9]+$).*$" # 阻擋純數字輸入
    if bool(re.fullmatch(pattern, message)) == False:
        return "請不要輸入亂碼"

    elif "推薦" in message:
        with driver as connector:
            with connector.session() as session:
                reponse = session.execute_read(get_recommendations_v2)

            completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "user",
                            "content": "請根據以下檢索到的內容來生成5個user可能提出的問題，回傳列點後的內容即可。\n檢索內容:%s" % reponse
                        }
                    ]
                )
            return completion.choices[0].message.content

    vector_search_start_time = time.perf_counter()
    openai_embedding = client.embeddings.create(
            model="text-embedding-3-large",
            input=message,
            encoding_format="float"
        ).data[0].embedding

    # chunk向量檢索(top 5)
    chunk_embedding_query = f"""
    SELECT (1 - (openai_embedding <=> '{str(openai_embedding)}')) AS similarity,text_segment,id
    FROM {PG_CHUNK_TABLE_NAME} 
    ORDER BY similarity 
    DESC limit 5;
    """
    cur.execute(chunk_embedding_query)
    vector_search_end_time = time.perf_counter()
    chunk_embedding_result = cur.fetchall()

    # 表格向量檢索(top 5)
    table_embedding_query = f"""
    SELECT (1 - (markdown_embedding <=> '{str(openai_embedding)}')) AS similarity,table_content,id
    FROM {PG_FORM_TABLE_NAME} 
    ORDER BY similarity 
    DESC limit 5;
    """
    cur.execute(table_embedding_query)
    vector_search_end_time = time.perf_counter()
    table_embedding_result = cur.fetchall()

    # 取text_segment/id
    chunk_embedding_top10 = [(item[1], item[2]) for item in chunk_embedding_result] 
    table_embedding_top10 = [(item[1], item[2]) for item in table_embedding_result] 

    print("向量搜尋資料筆數: ",len(chunk_embedding_top10),len(table_embedding_top10))
    print(f"\n向量搜尋執行時間: {vector_search_start_time-vector_search_end_time:.6f} 秒")
    
    # chunk全文檢索(top 5)
    fulltext_search_start_time = time.perf_counter()
    index_name = "chunk-index"
    search_keyword = message
    field_to_search = "text"
    search_body = {
        "query": {
            "match": {
            field_to_search: search_keyword
            }
        },
        "size": 5
    }
    chunk_response = es_client.search(index=index_name, body=search_body).body['hits']['hits']

    # table全文檢索(top 5)
    index_name = "table-index"
    search_keyword = message
    field_to_search = "text"
    search_body = {
        "query": {
            "match": {
            field_to_search: search_keyword
            }
        },
        "size": 5
    }
    table_response = es_client.search(index=index_name, body=search_body).body['hits']['hits']

    fulltext_search_end_time = time.perf_counter()
    chunk_fulltext_top20 = [item['_source']['text'] for item in chunk_response]
    table_fulltext_top20 = [item['_source']['text'] for item in table_response]
    print("全文檢索搜尋資料筆數: ",len(chunk_fulltext_top20),len(table_fulltext_top20))
    print(f"\n全文檢索執行時間: {fulltext_search_start_time-fulltext_search_end_time:.6f} 秒")

    # 檢索結果整理(drop duplicate)
    df_chunk_embedding_top10 = pd.DataFrame(chunk_embedding_top10,columns=['text_segment','id'])['text_segment']
    df_chunk_fulltext_top20 = pd.DataFrame(chunk_fulltext_top20,columns=['text_segment'])['text_segment']
    df_table_embedding_top10 = pd.DataFrame(table_embedding_top10,columns=['text_segment','id'])['text_segment']
    df_table_fulltext_top20 = pd.DataFrame(table_fulltext_top20,columns=['text_segment'])['text_segment']
    concat_result = pd.concat([df_chunk_fulltext_top20,df_table_fulltext_top20,df_chunk_embedding_top10,df_table_embedding_top10])
    print("df join前資料筆數",len(concat_result)) # 20
    concat_result = concat_result.drop_duplicates(keep='first').reset_index(drop=True).to_list() 
    print("df join後資料筆數: ",len(concat_result)) 
    
    # reranking結果，取10筆資料
    reranking_start_time = time.perf_counter()
    reranking = vo.rerank(query=message, documents=concat_result, model="rerank-2", top_k=10)
    reranking_end_time = time.perf_counter()
    reranking_result = [r.document for r in reranking.results]
    print("重排序後總參考資料筆數: ",len(reranking_result))
    print(f"\n全文檢索執行時間: {reranking_start_time-reranking_end_time:.6f} 秒")
    # reranking_result = concat_result
    
    # LLM輸出結果
    llm_start_time = time.perf_counter()
    completion = client.chat.completions.create(
        model="gpt-4.1",
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": "請根據以下檢索到的內容來回答user提出的問題，若不知道請回答不知道即可。\n檢索內容:%s" % reranking_result + "問題:%s" % message
            }
        ]
    )
    llm_end_time = time.perf_counter()
    print(f"\nLLM執行時間: {llm_start_time-llm_end_time:.6f} 秒")
    return completion.choices[0].message.content

page_1 = gr.ChatInterface(hybrid_search, type="messages", autofocus=False,description="向量搜尋搭配全文檢索後再經由LLM整合資訊")
demo = gr.TabbedInterface(interface_list=[page_1], tab_names=["混和搜尋"])

if __name__ == "__main__":
    demo.launch(debug=True)