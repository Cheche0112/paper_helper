import os
import sqlite3

from langchain_community.document_loaders import PDFPlumberLoader
from tqdm import tqdm

# 替换为硅基流动API
import requests
import json

# 设置硅基流动API密钥和URL
os.environ['SILICONFLOW_API_KEY'] = 'your_siliconflow_api_key_here'

# 硅基流动API调用函数
def call_siliconflow_api(messages, model="gpt-4"):
    """调用硅基流动API"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('SILICONFLOW_API_KEY')}"
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 2000
    }
    
    response = requests.post(
        "https://api.siliconflow.cn/v1/chat/completions",
        headers=headers,
        data=json.dumps(payload)
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API调用失败: {response.status_code}, {response.text}")

def chat_v3(full_text, questions):
    results = []

    for question in questions:
        """使用硅基流动大模型API分析文本"""
        sys_template = """
        你是一个专业文档分析助手，请根据以下文档内容回答问题。
        若文档中没有相关信息，请明确说明"根据文档内容无法回答该问题"。
        请根据问题要求给出简洁或者是具体详细的回答
        """

        user_question = f"""
        文档内容：{full_text[:30000]}  # 截断处理避免超出token限制
    
        问题：{question}
        
        请按问题要求直接给出答案
        """

        messages = [
            {"role": "system", "content": sys_template},
            {"role": "user", "content": user_question}
        ]
        
        try:
            # 调用硅基流动API
            response = call_siliconflow_api(messages, model="gpt-4")
            content = response["choices"][0]["message"]["content"]
            print(content)
            results.append(content)
        except Exception as e:
            print(f"API调用失败: {str(e)}")
            results.append("API调用失败，无法获取结果")
            
    return results


def update_analyse_database(questions, col_names, db_path: str = "papers.db"):
    """更新数据库中的PDF路径字段"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()


    try:
        # 检查并添加pdf列（如果不存在）
        cursor.execute("PRAGMA table_info(papers)")
        columns = [col[1] for col in cursor.fetchall()]

        for col in col_names:
            if col not in columns:
                cursor.execute(f"ALTER TABLE papers ADD COLUMN {col} TEXT")
                conn.commit()


        # 获取需要处理的记录
        cursor.execute(
            """
                SELECT title, pdf ,first_submitted_date
                FROM papers 
                WHERE question_detail IS NULL 
                  AND pdf is NOT NULL
                  AND abstract LIKE '%personality%'
                ORDER BY first_submitted_date DESC
            """
        )
        pending_papers = cursor.fetchall()

        # 添加进度条包装迭代对象
        for title, pdf, first_submitted_date in tqdm(pending_papers, desc="分析论文中", unit="篇"):
            try:
                tqdm.write(f"正在处理：{pdf}")  # 使用 tqdm.write 避免与进度条冲突
                pdf_path = pdf
                full_text = load_pdf_content(pdf_path)
                results = chat_v3(full_text, questions)
            except Exception as e:
                continue

            try:
                for i in range(len(results)):
                    # 更新数据库记录
                    cursor.execute(
                        f"UPDATE papers SET {col_names[i]} = ? WHERE title = ?",
                        (f"{results[i]}", title)
                    )
                    conn.commit()
                tqdm.write(f"成功更新：{pdf_path}")

            except Exception as e:
                tqdm.write(f"处理失败 [{pdf}]: {str(e)}")
                conn.rollback()
                continue


    finally:
        conn.close()


def load_pdf_content(file_path):
    """加载并返回PDF全文内容"""
    try:
        loader = PDFPlumberLoader(file_path)
        documents = loader.load()
        return "\n".join([doc.page_content for doc in documents])
    except Exception as e:
        print(f"加载PDF失败: {e}")
        return None




if __name__ == "__main__":
    # 示例问题
    questions = [
        """
        论文标题和标题的中文翻译是什么？请按以下格式分别直接给出:
        英文: [英文标题]
        中文: [标题中文翻译]
        """
        ,

        """
        论文摘要和你对摘要的中文翻译是什么？请严格按以下格式分别完整地给出:
        英文: [英文摘要]
        中文: [摘要的中文翻译]
        """
        ,

        """
        文档的主要研究内容是什么？回答这个问题是可相对详细一些，并侧重关注和考虑分析是否包含有"人格、性格(personality)"的部分,前提是有对应的内容,请你按以下格式输出：
        主要内容: [文本描述]
        关于人格、性格: [文本描述]
        
        请注意，关于人格、性格的内容如果没有，则标为"无"
        """

        ,
        """
        论文使用的数据集有哪些？如果没有，则回答"无"，如果有请严格按以下格式列出：
        数据集: [数据集名称]
        数据集规模: [数据集规模]
        数据集形式: [数据集形式]
        地址: [数据集url]
        
        如果上述格式对应的项（规模、形式、地址）没有，则标为"无"，请注意数据集名称不能为"无"
        """
    ]

    col_names = [
        "question_title",
        "question_abs",
        "question_detail",
        "question_dataset"
    ]

    update_analyse_database(questions, col_names)





