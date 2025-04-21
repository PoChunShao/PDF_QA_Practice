from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger
import sys

ws_driver = None 
pos_driver = None

try:
    print("Initializing CKIP WS Driver...")
    ws_driver  = CkipWordSegmenter(model_name="../bert-base-chinese-ws")
    print("Initializing CKIP POS Driver...")
    pos_driver = CkipPosTagger(model_name="../bert-base-chinese-pos")
    print("CKIP Drivers initialized successfully.")
except Exception as e:
    print(f"Error initializing CKIP drivers: {e}", file=sys.stderr)

def clean(sentence_ws, sentence_pos):
    short_with_pos = []
    short_sentence = []
    stop_pos = set(['Nep', 'Nh', "Nb"]) # 這 3 種詞性不保留

    for word_ws, word_pos in zip(sentence_ws, sentence_pos):
        # 只留名詞和動詞和外文
        is_N_or_V = word_pos.startswith("V") or word_pos.startswith("N") or word_pos.startswith("FW")

        # 去掉名詞裡的某些詞性
        is_not_stop_pos = word_pos not in stop_pos

        # 只剩一個字的詞也不留
        is_not_one_charactor = not (len(word_ws) == 1)

        # 組成串列
        if is_N_or_V and is_not_stop_pos and is_not_one_charactor:
            short_with_pos.append(f"{word_ws}({word_pos})")
            short_sentence.append(f"{word_ws}")

    return short_sentence, short_with_pos

def get_clean_output(text):
    ws = ws_driver(text)
    pos = pos_driver(ws)
    for sentence_ws, sentence_pos in zip(ws, pos):
        short, res = clean(sentence_ws, sentence_pos)
    return short