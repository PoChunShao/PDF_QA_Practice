from ckip_transformers.nlp import CkipWordSegmenter
from rank_bm25 import BM25Okapi
import sys

ws_driver = None 
pos_driver = None

try:
    print("Initializing CKIP WS Driver...")
    ws_driver  = CkipWordSegmenter(model_name="../bert-base-chinese-ws")
    print("CKIP Drivers initialized successfully.")
except Exception as e:
    print(f"Error initializing CKIP drivers: {e}", file=sys.stderr)

def get_bm25_result(n=10,text_ws_list=None,query_ws_list=None):
    """針對like關鍵字搜尋結果，使用bm25進行二次排序"""
    tokenized_corpus  = ws_driver(text_ws_list)
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_top10 = bm25.get_top_n(query_ws_list, text_ws_list, n)
    # print(bm25_top10)
    return bm25_top10