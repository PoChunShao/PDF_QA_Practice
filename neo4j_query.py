from random import sample

basic_node_labels = [
    "Project",              # 代表具體的計畫，例如本計畫 "112DA023"
    "Organization",         # 代表組織或單位，如政府機關、大學、公司、國際組織等
    "Person",               # 代表參與計畫或被提及的個人 (不論是具名或角色)
    "SystemTool",           # 代表系統、軟體工具、模型、計算方法等，如 SAS, QSAR Toolbox, TEST
    "DataSource",           # 代表資料的來源，如資料庫、清單、報告、文章、網站等
    "ChemicalSubstance",    # 代表化學物質，可以是具體物質 (如 PFAS, 苯) 或物質類別
    "Concept",              # 代表抽象概念、方法學、屬性、危害類別等，如綠色化學, QSAR, 危害終點, CAS No.
    "Event",                # 代表發生的事件，如會議、教育訓練、工作坊等
    "Location",             # 代表地理位置或區域，如臺灣, 美國, 歐盟
    "Regulation",           # 代表法規、標準或指南，如 REACH, 加州 65 號提案
    "Keyword"
]

def get_recommendations(tx):
    '''隨機選取chunk節點，並將透過NEXT_CHUNK連接的節點中的text屬性取出，作為生成問題的依據'''
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

def get_recommendations_v2(tx,label_list=basic_node_labels):
    '''隨機選取topic類別的節點，並節點中所有的關聯及的name,text屬性取出，作為生成問題的依據'''
    random_topic_list = sample(label_list,3)
    topic_1 = random_topic_list[0]
    topic_2 = random_topic_list[1]
    topic_3 = random_topic_list[2]

    query_1 = f"""
        MATCH (p:{topic_1}) WITH p ORDER BY rand() LIMIT 1
        MATCH (c:{topic_2}) WITH p, c ORDER BY rand() LIMIT 1
        MATCH (proj:{topic_3}) WITH p, c, proj ORDER BY rand() LIMIT 1
    """
    query_2 = """
        OPTIONAL MATCH (selected_node)-[r]-(external_node)
        WHERE selected_node IN [p, c, proj] AND NOT external_node IN [p, c, proj]
        RETURN p, c, proj,
            collect(DISTINCT {
                source: selected_node,
                relationship: r,
                external_neighbor: external_node,
                external_neighbor_properties: properties(external_node)
            }) AS external_connections_with_properties
    """
    query = query_1 + query_2
    result = tx.run(query).single()

    if result:
        connections_list = result["external_connections_with_properties"]
        final_result = []

        for i, connection_info in enumerate(connections_list):
            # a) 取得來源節點物件 (Node)
            source_node = connection_info["source"]
            source_id = source_node.id
            # source_labels = source_node.labels
            labels_dict = dict(connection_info.items())['external_neighbor'].labels
            labels_to_exclude = {'__Entity__', '__KGBuilder__'}
            source_labels = set(labels_dict)-labels_to_exclude

            # b) 取得關係物件 (Relationship)
            relationship = connection_info["relationship"]
            relationship_type = relationship.type
            relationship_props = dict(relationship.items()) # 關係的屬性

            # c) 取得外部鄰居節點物件 (Node)
            neighbor_node = connection_info["external_neighbor"]
            neighbor_id = neighbor_node.id

            # d) 取得外部鄰居的屬性 Map (這已經是 Python 字典)
            neighbor_properties = connection_info["external_neighbor_properties"]

            # ==================== 修改開始 ====================
            relevant_neighbor_prop_name = None
            relevant_neighbor_prop_value = None

            # 檢查 relationship_type
            if relationship_type == "FROM_CHUNK" or relationship_type == "NEXT_CHUNK":
                relevant_neighbor_prop_name = "text"
                # 從 neighbor_properties 字典中安全地取得 'text' 屬性
                relevant_neighbor_prop_value = neighbor_properties.get(relevant_neighbor_prop_name, "N/A")

            else:
                # 對於所有其他 relationship_type
                relevant_neighbor_prop_name = "name"
                # 從 neighbor_properties 字典中安全地取得 'name' 屬性
                relevant_neighbor_prop_value = neighbor_properties.get(relevant_neighbor_prop_name, "N/A")

            node_info = {
                "source_labels" : source_labels,
                "source_name" : source_node.get('name',source_node.get('text')),
                "relationship_type":relationship_type,
                "relationship_detail":relationship_props,
                "relevant_neighbor_prop_name":relevant_neighbor_prop_name,
                "relevant_neighbor_prop_value":relevant_neighbor_prop_value
            }

            final_result.append(node_info)

        return final_result