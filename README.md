# PDF_QA_Practice
## 前情提要
* pdf sample: https://www.cha.gov.tw/sp-research-cont-345-09033-1.html
* 本次練習題以gradio展示，請以app_v2.py為主
* bm25.py及wordsegment.py為實作postgresql全文檢索的模組，但因後續切轉Elaticsearch後棄用
* custom.dic為elaticsearch額外替換的繁中字典
* 地端服務有3個container(elaticsearch,kibana,postgresql)，neo4j則採用AURA雲端版
