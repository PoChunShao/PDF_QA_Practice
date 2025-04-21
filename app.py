import neo4j
import gradio as gr
import psycopg2
import time
import re
import pandas as pd
import voyageai

from openai import OpenAI
from dotenv import load_dotenv
from wordsegment import get_clean_output
from bm25 import get_bm25_result
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings

# voyage rerank config
vo = voyageai.Client(api_key="pa-l4awt3oZKjHdCZqJsYcfYLUYdXkpUv103J8nMbP3ubq")

# openai config
load_dotenv('.env', override=True)
client = OpenAI(api_key="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")

# neo4j config
NEO4J_URI="neo4j+ssc://8f44408d.databases.neo4j.io"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="3nct06kMQOSlmFi0lRYfpfeXj4Jjy47_cq-YHueeVxA"
driver = neo4j.GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
embedder = OpenAIEmbeddings(model="text-embedding-3-large")

# postgresql config
endpoint = "localhost"
database = "postgres"
username = "postgres"
password = "mysecretpassword"
port = 8080
connection = psycopg2.connect(
    dbname=database,
    host=endpoint,
    port=port,
    user=username,
    password=password
)

cur = connection.cursor()
TABLE_NAME = "public.section_content_charsplitt"

def hybrid_search(message,history):
    pattern = r"^(?![0-9]+$).*$" # 阻擋純數字輸入
    if bool(re.fullmatch(pattern, message)) == False:
        return "請不要輸入亂碼"
    elif message=="推薦":
        with neo4j.GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as connector:
            def get_all_nodes(tx):
                temp = []
                result = tx.run("""
                            MATCH (c:Chunk)-[:NEXT_CHUNK]->(c2:Chunk)-[:NEXT_CHUNK]->(c3:Chunk)-[:NEXT_CHUNK]->(c4:Chunk)
                            // 2. 返回這 4 個節點各自的 text 屬性
                            RETURN c.text AS chunk1_text,
                                c2.text AS chunk2_text,
                                c3.text AS chunk3_text,
                                c4.text AS chunk4_text
                            ORDER BY rand() // 3. 隨機排序找到的所有路徑
                            LIMIT 1         // 4. 只取隨機排序後的第一個路徑結果
                        """)
                for record in result:
                    temp.append(record)
                return temp 
            
            with connector.session() as session:
                reponse = session.read_transaction(get_all_nodes)

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

    # 向量搜尋
    vector_search_start_time = time.perf_counter()
    openai_embedding = client.embeddings.create(
            model="text-embedding-3-large",
            input=message,
            encoding_format="float"
        ).data[0].embedding

    embedding_query = f"""
    SELECT (1 - (openai_embedding <=> '{str(openai_embedding)}')) AS similarity,text_segment,id
    FROM {TABLE_NAME} 
    ORDER BY similarity 
    DESC limit 10;
    """
    # chunk向量檢索(top 10)
    cur.execute(embedding_query)
    vector_search_end_time = time.perf_counter()
    embedding_result = cur.fetchall()
    
    embedding_top10 = [(item[1], item[2]) for item in embedding_result] # 取text_segment/id
    # embedding_top10 = [i[1] for i in embedding_result]
    print("向量搜尋資料筆數: ",len(embedding_top10))
    print(f"\n向量搜尋執行時間: {vector_search_start_time-vector_search_end_time:.6f} 秒")
    
    # 全文檢索(top 20)
    fulltext_search_start_time = time.perf_counter()
    words = get_clean_output([message])

    # 生成 CASE WHEN 子句列表，將 CASE WHEN 子句用 '+' 連接
    case_clauses = [f"CASE WHEN text_segment LIKE '%{word}%' THEN 1 ELSE 0 END" for word in words]
    score_calculation = " +\n      ".join(case_clauses) # 加入換行和縮排以提高可讀性

    # 生成原始的 OR 條件用於 WHERE 子句
    where_conditions = [f"text_segment LIKE '%{word}%'" for word in words]
    where_clause = " OR\n    ".join(where_conditions) # 加入換行和縮排

    sql_query = f"""
    SELECT
        text_segment,id,
        ( -- 計算命中分數開始
        {score_calculation}
        ) AS match_count -- 計算命中分數結束
    FROM
        {TABLE_NAME}
    WHERE
        {where_clause}
    ORDER BY
        match_count DESC
    LIMIT 20;
    """
    cur.execute(sql_query)
    fulltext_search_end_time = time.perf_counter()
    fulltext_result = cur.fetchall()
    fulltext_top20 = [i[0] for i in fulltext_result]
    print("全文檢索搜尋資料筆數: ",len(fulltext_top20))
    print(f"\n全文檢索執行時間: {fulltext_search_start_time-fulltext_search_end_time:.6f} 秒")

    # 針對like關鍵字搜尋結果，使用bm25進行二次排序
    bm25_strat_time = time.perf_counter()
    bm25_top10 = get_bm25_result(n=10,text_ws_list=fulltext_top20,query_ws_list=words)
    bm25_end_time = time.perf_counter()
    print("BM25資料筆數: ",len(bm25_top10))
    print(f"\nBM25執行時間: {bm25_strat_time-bm25_end_time:.6f} 秒")

    # 檢索結果整理(drop duplicate)
    embedding_top10 = pd.DataFrame(embedding_top10,columns=['text_segment','id'])['text_segment']
    bm25_top10 = pd.DataFrame(bm25_top10,columns=['text_segment'])['text_segment']
    concat_result = pd.concat([bm25_top10,embedding_top10]).drop_duplicates(keep='first').reset_index(drop=True).to_list()
    print("df合併後總參考資料筆數: ",len(concat_result))

    # reranking合併後結果，取10筆資料
    reranking = vo.rerank(query=message, documents=concat_result, model="rerank-2", top_k=10)
    reranking_result = [r.document for r in reranking.results]
    print("重排序後總參考資料筆數: ",len(reranking_result))

    # LLM輸出結果
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": "請根據以下檢索到的內容來回答user提出的問題，若不知道請回答不知道即可。\n檢索內容:%s" % reranking_result + "問題:%s" % message
            }
        ]
    )
    return completion.choices[0].message.content

def query_table(message,history):
    # 阻擋純數字輸入
    pattern = r"^(?![0-9]+$).*$"
    if bool(re.fullmatch(pattern, message)) == False:
        return "請不要輸入亂碼"
    
    vector_search_start_time = time.perf_counter()
    # 向量搜尋
    openai_embedding = client.embeddings.create(
            model="text-embedding-3-large",
            input=message,
            encoding_format="float"
        ).data[0].embedding

    embedding_query = f"""
    SELECT (1 - (markdown_embedding <=> '{str(openai_embedding)}')) AS similarity,table_content 
    FROM public.table_markdown_v2
    ORDER BY similarity 
    DESC limit 3;
    """

    # 向量檢索(top 3)
    cur.execute(embedding_query)
    vector_search_end_time = time.perf_counter()
    embedding_result = cur.fetchall()
    enbedding_top10 = [i[1] for i in embedding_result]
    # print(enbedding_top10)
    print(f"\n向量搜尋執行時間: {vector_search_start_time-vector_search_end_time:.6f} 秒")
    
    # 全文檢索(top 10)
    query_segment_start_time = time.perf_counter()
    words = get_clean_output([message])
    query_segment_end_time = time.perf_counter()
    print(f"\n問句斷詞執行時間: {query_segment_start_time-query_segment_end_time:.6f} 秒")

    fulltext_search_start_time = time.perf_counter()
    # 生成 CASE WHEN 子句列表
    case_clauses = [f"CASE WHEN table_content LIKE '%{word}%' THEN 1 ELSE 0 END" for word in words]
    # 將 CASE WHEN 子句用 '+' 連接
    score_calculation = " +\n      ".join(case_clauses) # 加入換行和縮排以提高可讀性

    # 生成原始的 OR 條件用於 WHERE 子句
    where_conditions = [f"table_content LIKE '%{word}%'" for word in words]
    where_clause = " OR\n    ".join(where_conditions) # 加入換行和縮排

    sql_query = f"""
    SELECT
        table_content,
        ( -- 計算命中分數開始
        {score_calculation}
        ) AS match_count -- 計算命中分數結束
    FROM
        public.table_markdown_v2
    WHERE
        {where_clause}
    ORDER BY
        match_count DESC
    LIMIT 5;
    """
    cur.execute(sql_query)
    fulltext_search_end_time = time.perf_counter()
    fulltext_result = cur.fetchall()
    fulltext_top20 = [i[0] for i in fulltext_result]
    print(len(fulltext_top20))
    print(f"\n全文檢索執行時間: {fulltext_search_start_time-fulltext_search_end_time:.6f} 秒")

    # 針對like關鍵字搜尋結果，使用bm25進行二次排序
    bm25_start_time = time.perf_counter()
    bm25_top10 = get_bm25_result(n=3,text_ws_list=fulltext_top20,query_ws_list=words)
    bm25_end_time = time.perf_counter()
    print(f"\nBM25執行時間: {bm25_start_time-bm25_end_time:.6f} 秒")

    # 檢索結果整理
    llm_reference = []
    llm_reference.append([enbedding_top10,bm25_top10])
    openai_start_time = time.perf_counter()
    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "user",
                "content": "請根據以下檢索到的markdown表格內容來回答user提出的問題，並在最後附上檢索到表格，若不知道請回答不知道即可。\n檢索內容:%s" % llm_reference + "問題:%s" % message
            }
        ]
    )
    openai_end_time = time.perf_counter()
    print(f"\nopenai執行時間: {openai_start_time-openai_end_time:.6f} 秒")
    return completion.choices[0].message.content


with neo4j.GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as connector:
    def generate_question(message,history):

        def get_all_nodes(tx):
            temp = []
            result = tx.run("""
                        MATCH (c:Chunk)-[:NEXT_CHUNK]->(c2:Chunk)-[:NEXT_CHUNK]->(c3:Chunk)-[:NEXT_CHUNK]->(c4:Chunk)
                        // 2. 返回這 4 個節點各自的 text 屬性
                        RETURN c.text AS chunk1_text,
                            c2.text AS chunk2_text,
                            c3.text AS chunk3_text,
                            c4.text AS chunk4_text
                        ORDER BY rand() // 3. 隨機排序找到的所有路徑
                        LIMIT 1         // 4. 只取隨機排序後的第一個路徑結果
                    """)
            for record in result:
                temp.append(record)
            return temp 
        
        with connector.session() as session:
            reponse = session.read_transaction(get_all_nodes)

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

page_1 = gr.ChatInterface(hybrid_search, type="messages", autofocus=False,description="主要以向量搜尋搭配全文檢索+BM25取得多樣化的檢索結果")
page_2 = gr.ChatInterface(query_table, type="messages", autofocus=False, description="檢索表格中的內容並回傳markdown表格")
page_3 = gr.ChatInterface(generate_question, type="messages", autofocus=False,description="根據知識圖譜生成推薦問題")
demo = gr.TabbedInterface(interface_list=[page_1, page_2, page_3], tab_names=["混和搜尋", "表格檢索", "生成推薦問題"])

if __name__ == "__main__":
    demo.launch(debug=True)



# def page1_function(input_text):
#     return f"Page 1 output: {input_text}"

# def page2_function(input_number):
#     return f"Page 2 output: {input_number ** 2}"

# page1 = gr.Interface(fn=page1_function, inputs="text", outputs="text", title="Page 1")
# page2 = gr.Interface(fn=page2_function, inputs="number", outputs="text", title="Page 2")

# # 使用 TabbedInterface 创建分頁
# app = gr.TabbedInterface([page1, page2], ["Page 1", "Page 2"])

# # 启动应用
# app.launch()