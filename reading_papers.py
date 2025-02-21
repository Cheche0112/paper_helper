import os
import sqlite3
import json
import logging
from datetime import datetime
from tqdm import tqdm
from langchain_community.document_loaders import PDFPlumberLoader
from volcenginesdkarkruntime import Ark

# 配置增强日志系统
# 修改日志配置部分
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("paper_analysis.log", encoding='utf-8'),  # 添加编码参数
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 其他原有配置保持不变
os.environ['ARK_API_KEY'] = '89f37545-ebf9-45b4-a537-fb281f2204c9'

client = Ark(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key=os.environ.get("ARK_API_KEY")
)


def chat_v3(full_text, questions):
    results = []
    for question in questions:
        try:
            sys_template = """你是一个专业文档分析助手，请根据以下文档内容回答问题。
            若文档中没有相关信息，请明确说明"根据文档内容无法回答该问题"。
            请根据问题要求给出简洁或者是具体详细的回答"""

            user_question = f"""文档内容：{full_text[:33000]} 
            问题：{question}
            请按问题要求直接给出答案"""

            completion = client.chat.completions.create(
                model="ep-20250206224347-2mshv",
                messages=[
                    {"role": "system", "content": sys_template},
                    {"role": "user", "content": user_question},
                ],
            )
            result = completion.choices[0].message.content
            results.append(result)
            logger.info(f"成功处理问题: {question[:30]}...")  # 截断长问题
        except Exception as e:
            logger.error(f"处理问题时出错: {str(e)}")
            results.append("ERROR: 处理失败")
    return results





# 其他配置保持不变...

def update_analyse_database(questions, col_names, db_path: str = "papers.db"):
    """更新数据库并实时生成JSON输出"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 生成唯一结果文件名
    output_file = f"results/analysis_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
    os.makedirs("results", exist_ok=True)

    all_results = []
    stats = {
        "total": 0,
        "success": 0,
        "failed": 0,
        "current_batch": 0
    }

    try:
        # ... [原有表结构检查代码保持不变] ...
        # 检查并添加pdf列（如果不存在）
        cursor.execute("PRAGMA table_info(papers)")
        columns = [col[1] for col in cursor.fetchall()]

        for col in col_names:
            if col not in columns:
                cursor.execute(f"ALTER TABLE papers ADD COLUMN {col} TEXT")
                conn.commit()

        # 获取待处理论文
        cursor.execute("""
            SELECT title, pdf, first_submitted_date 
            FROM papers 
            WHERE question_detail IS NULL 
              AND pdf is NOT NULL
              AND abstract LIKE '%personality%'
            ORDER BY first_submitted_date DESC
        """)
        pending_papers = cursor.fetchall()
        stats["total"] = len(pending_papers)

        for idx, (title, pdf, date) in enumerate(tqdm(pending_papers, desc="分析论文"), 1):
            paper_data = {
                "meta": {
                    "title": title,
                    "pdf_path": pdf,
                    "submit_date": date,
                    "process_time": datetime.now().isoformat(),
                    "status": "pending"
                },
                "answers": {}
            }

            try:
                # PDF内容加载
                full_text = load_pdf_content(pdf)
                if not full_text:
                    logger.warning(f"🟡 内容空值跳过 | {title}")
                    stats["failed"] += 1
                    continue

                # 大模型分析
                results = chat_v3(full_text, questions)

                # 构建数据结构
                for col_name, result in zip(col_names, results):
                    paper_data["answers"][col_name] = result

                # 更新数据库
                update_params = [*results, title]
                cursor.execute(
                    f"UPDATE papers SET {', '.join([f'{col}=?' for col in col_names])} WHERE title = ?",
                    update_params
                )
                conn.commit()

                # 更新结果集
                paper_data["meta"]["status"] = "success"
                all_results.append(paper_data)
                stats["success"] += 1

                # 实时写入JSON
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, ensure_ascii=False, indent=2)

                # 记录论文摘要日志
                logger.info(f"\n🔵 论文分析完成 [{idx}/{stats['total']}]")
                logger.info(f"  标题：{title}")
                logger.info(f"  状态：✅ 成功入库")
                logger.info(f"  关键结果：")
                for q_name in ["question_title", "question_dataset"]:
                    if ans := paper_data["answers"].get(q_name):
                        logger.info(f"  - {q_name}: {ans[:80]}...")

            except Exception as e:
                paper_data["meta"]["status"] = f"failed: {str(e)}"
                all_results.append(paper_data)
                stats["failed"] += 1
                logger.error(f"🔴 处理失败 | {title}\n{str(e)}", exc_info=True)
                conn.rollback()

            # 阶段性进度报告
            if idx % 5 == 0 or idx == stats["total"]:
                progress = f"\n📊 处理进度 [{idx}/{stats['total']}]\n"
                progress += f"✅ 成功：{stats['success']} 篇\n"
                progress += f"❌ 失败：{stats['failed']} 篇\n"
                progress += f"📂 最新结果文件：{output_file}"
                logger.info(progress)

                # 更新实时结果文件
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, ensure_ascii=False, indent=2)

        # 最终总结报告
        final_report = f"\n🎉 分析任务完成！\n"
        final_report += f"➤ 共处理论文：{stats['total']} 篇\n"
        final_report += f"➤ 成功分析数：{stats['success']} 篇 ({stats['success'] / stats['total']:.1%})\n"
        final_report += f"➤ 失败分析数：{stats['failed']} 篇\n"
        final_report += f"➤ 结果文件路径：{os.path.abspath(output_file)}"
        logger.info(final_report)

    finally:
        conn.close()

def load_pdf_content(file_path):
    """加载并返回PDF全文内容"""
    try:
        loader = PDFPlumberLoader(file_path)
        documents = loader.load()
        content = "\n".join([doc.page_content for doc in documents])
        logger.info(f"成功加载PDF: {file_path} [{len(content)}字符]")
        return content
    except Exception as e:
        logger.error(f"加载PDF失败: {file_path} - {str(e)}")
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
        文档的主要研究内容是什么？回答这个问题是可相对详细一些，并侧重关注和考虑分析是否包含有“人格、性格(personality)”的部分,前提是有对应的内容,请你按以下格式输出：
        主要内容: [文本描述]
        关于人格、性格: [文本描述]

        请注意，关于人格、性格的内容如果没有，则标为“无”
        """

        ,
        """
        论文使用的数据集有哪些？如果没有，则回答“无”，如果有请严格按以下格式列出：
        数据集: [数据集名称]
        数据集规模: [数据集规模]
        数据集形式: [数据集形式]
        地址: [数据集url]

        如果上述格式对应的项（规模、形式、地址）没有，则标为“无”，请注意数据集名称不能为“无”
        """
    ]

    col_names = [
        "question_title",
        "question_abs",
        "question_detail",
        "question_dataset"
    ]

    update_analyse_database(questions, col_names)