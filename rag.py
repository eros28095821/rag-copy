import pandas as pd
from langchain_ollama import ChatOllama
# 读取 Excel 文件并加载案件参考数据
def load_reference_cases(file_path, max_cases=5, max_chars_per_case=200):
    try:
        df = pd.read_excel(file_path, sheet_name='判決書550筆ver1')  # 使用文件路径读取文件
        reference_cases = []
        for index, row in df.iterrows():
            # 创建案件摘要并限制字符数
            case_summary = f"案件{index + 1}：原告{row.get('Unnamed: 5', '未知原告')}，被告{row.get('Unnamed: 6', '未知被告')}，事故緣由：{row.get('Unnamed: 7', '未知事故緣由')}。"
            reference_cases.append(case_summary[:max_chars_per_case])  # 截取到最大字符数
            if len(reference_cases) >= max_cases:
                break  # 限制总数
        return reference_cases
    except Exception as e:
        print("读取 Excel 文件时出错：", e)
        return []
 # 初始化模型
llm = ChatOllama(
    model="kenneth85/llama-3-taiwan:8b-instruct",
    temperature=0.1,
    #num_predict=256,
)
# 手动输入新案件的数据，包括靈活的賠償項目
def manual_input_case_data():
    case_data = {}
    case_data['原告'] = input("請輸入原告姓名：")
    case_data['被告'] = input("請輸入被告姓名：")
    case_data['事故緣由'] = input("請輸入事故發生緣由：")
    
    # 賠償項目
    case_data['賠償項目'] = {}
    case_data['賠償項目']['醫藥費用'] = int(input("請輸入醫藥費用（元）：").replace(',', '') or "0")
    case_data['賠償項目']['看護費用'] = int(input("請輸入看護費用（元）：").replace(',', '') or "0")
    case_data['賠償項目']['喪失工作所得'] = int(input("請輸入喪失工作所得（元）：").replace(',', '') or "0")
    case_data['賠償項目']['精神慰撫金'] = int(input("請輸入精神慰撫金（元）：").replace(',', '') or "0")
    
    # 動態添加其他賠償項目
    while True:
        add_more = input("是否要添加其他賠償項目？(y/n)：")
        if add_more.lower() == 'y':
            item_name = input("請輸入賠償項目名稱：")
            item_amount = int(input(f"請輸入{item_name}金額（元）：").replace(',', '') or "0")
            case_data['賠償項目'][item_name] = item_amount
        else:
            break
    
    # 計算總金額
    case_data['總金額'] = sum(case_data['賠償項目'].values())
    
    return case_data

# 生成律师函
def generate_lawyer_letter(reference_cases, case_data):
    # 將參考案件集合為單個字符串
    reference_text = "\n".join(reference_cases)
    
    # 構建賠償項目細節文本
    compensation_details = "\n".join([f"    - {item}：{amount}元" for item, amount in case_data['賠償項目'].items()])
    
    case_exp= """
    範例案件1：在交通事故中，原告因被告駕駛車輛不當操作而受傷，引用第184條和第193條作為侵權賠償條款。
    範例案件2：多人共同造成原告受傷，但無法確定具體加害人時，引用第185條作為連帶賠償依據。
    範例案件3：被告駕駛機車行駛途中疏忽駕駛，撞傷行人，依第191-2條駕駛人應負賠償責任。
    範例案件4：被告在營業性運輸活動中操作失當，導致原告受傷，依第191-3條被告應負特別損害賠償責任。
    範例案件5：原告因被告在交通事故中造成重傷，依第193條請求其醫療費、看護費等損害賠償。
    範例案件6：原告因被告的過失行為遭受嚴重精神創傷，依第195條請求精神慰撫金。
    範例案件7：原告的車輛在交通事故中損壞，依第217條請求財產損害賠償。
    範例案件8：被告酒駕並肇事，原告引用《道路交通管理處罰條例》第62條以加強賠償依據。
    範例案件9：被告因超速駕駛導致車禍，原告引用《道路交通管理處罰條例》第43條，以證明被告違規行為。
    範例案件10：被告駕駛車輛未遵守安全駕駛規範，違反《道路交通管理處罰條例》第82條，原告引用此條款作為賠償依據。
    """
    # 构建生成律师函的 Prompt
    prompt = f"""
    以下是一些已審結的案件摘要，可作為參考依據：
    {reference_text}{case_exp}

    根據以下新案件資料，生成一份完整的民事交通事故起訴狀草稿，包括詳細的事實描述、法律條款引用和賠償請求，格式要求如下：

    新案件資料：
    - 原告：{case_data['原告']}
    - 被告：{case_data['被告']}
    - 事故經過：{case_data['事故緣由']}
    - 賠償費用明細：
    {compensation_details}
    - 賠償費用總金額：{case_data['總金額']}元
    
    起訴狀格式應包括：
    
    一、事實緣由：
    描述事故發生的經過。格式需完整且符合法律文件的正式語氣，清晰指出原告如何遭受傷害，以及該傷害的嚴重程度。例如，詳細描述被告如何基於故意或過失，對原告造成了什麼具體的身體或精神傷害。

    二、被害結果：
    詳細描述原告受傷或財產損失情況，包含醫療需求、治療過程及後遺症。

    三、損害賠償的事實及金額：分別列出醫療費用、看護費用、工作損失、精神慰撫金等，並計算總金額。
    
    四、引用法律條款：
    明確列出相關的法律條款，包括《中華民國民法》第184條、第185條、第193條和第195條，並根據案件的情況說明該條款如何適用於賠償請求。請以分段形式呈現各條款，例如：「因故意或過失，不法侵害他人之權利者，負損害賠償責任。」。並引用每條法律條款的適用情況。

    五、賠償請求：
    列出以上賠償項目及其金額，使用詳細說明以表達合理性，且請求被告支付賠償金額及利息。
    - 合計賠償金額：{case_data['總金額']}元
    
    以上結構應嚴謹且正式，遵循法律術語的使用規範。
    """
  
# 將 prompt 作為參數傳遞並直接生成結果
    try:
        # 使用 invoke 方法並包裝 prompt
        response = llm.invoke([{"role": "user", "content": prompt}])
        
        # 提取 content 並將 \n 替換為真正的換行符號
        lawyer_letter = response.content.replace("\\n", "\n")
        
        return lawyer_letter
    except Exception as e:
        print("生成律師函時出錯：", e)
        return None

# 主程序逻辑
if __name__ == "__main__":
    # 使用文件路径加载 Excel 参考数据，限制为5个案件
    file_path = '/home/chen/rag-copy/起訴狀案例測試.xlsx'
    reference_cases = load_reference_cases(file_path, max_cases=5, max_chars_per_case=200)
    
    # 手动输入新案件的資料
    print("請手動輸入新案件的資料：")
    case_data = manual_input_case_data()
    
    # 生成律师函
    lawyer_letter = generate_lawyer_letter(reference_cases, case_data)
    
    # 打印生成的律师函
    print("\n生成的律师函：\n", lawyer_letter)
