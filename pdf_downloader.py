import os
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from urllib.parse import urlparse

import requests
from tqdm import tqdm


# 更新Paper类以包含pdf字段
@dataclass
class Paper:
    first_submitted_date: datetime
    title: str
    categories: List[str]
    url: str
    authors: str
    abstract: str
    comments: str
    title_translated: Optional[str] = None
    abstract_translated: Optional[str] = None
    first_announced_date: Optional[datetime] = None
    pdf: Optional[str] = None  # 新增PDF路径字段


def get_arxiv_pdf(url: str, pdf_dir: str = "./pdfs") -> str:
    """下载arXiv论文PDF并返回保存路径"""
    # 解析URL
    parsed_url = urlparse(url)
    if "/abs/" not in parsed_url.path:
        raise ValueError("Invalid arXiv abstract page URL")

    # 提取并清理arXiv ID
    arxiv_id = parsed_url.path.split("/")[-1]
    base_id = re.sub(r"v\d+$", "", arxiv_id)  # 移除版本号
    filename = f"{base_id.replace('/', '_')}.pdf"
    os.makedirs(pdf_dir, exist_ok=True)
    output_path = os.path.join(pdf_dir, filename)

    # 如果文件已存在，直接返回路径
    if os.path.exists(output_path):
        return os.path.abspath(output_path)

    # 构造PDF URL
    pdf_url = f"https://arxiv.org/pdf/{base_id}.pdf"

    # 添加浏览器头防止被拒绝
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }

    # 下载PDF
    response = requests.get(pdf_url, headers=headers, timeout=30)
    if response.status_code != 200:
        raise Exception(f"下载失败，状态码：{response.status_code}")

    # 保存文件
    with open(output_path, "wb") as f:
        f.write(response.content)

    return os.path.abspath(output_path)


def update_pdf_database(db_path: str = "papers.db", pdf_dir: str = "./pdfs"):
    """更新数据库中的PDF路径字段"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # 检查并添加pdf列（如果不存在）
        cursor.execute("PRAGMA table_info(papers)")
        columns = [col[1] for col in cursor.fetchall()]
        if "pdf" not in columns:
            cursor.execute("ALTER TABLE papers ADD COLUMN pdf TEXT")
            conn.commit()

        # 获取需要处理的记录
        cursor.execute("SELECT title, url FROM papers WHERE pdf IS NULL and abstract LIKE '%personality%'")
        pending_papers = cursor.fetchall()

        # 添加进度条包装迭代对象
        for title, url in tqdm(pending_papers, desc="下载论文", unit="篇"):
            try:
                tqdm.write(f"正在处理：{url}")  # 使用 tqdm.write 避免与进度条冲突
                pdf_path = get_arxiv_pdf(url, pdf_dir)

                # 更新数据库记录
                cursor.execute(
                    "UPDATE papers SET pdf = ? WHERE title = ?",
                    (pdf_path, title)
                )
                conn.commit()
                tqdm.write(f"成功更新：{pdf_path}")

                # 礼貌性延迟防止被封
                time.sleep(1)

            except Exception as e:
                tqdm.write(f"处理失败 [{url}]: {str(e)}")
                conn.rollback()
                continue

    finally:
        conn.close()


if __name__ == "__main__":
    # 示例用法
    update_pdf_database(
        db_path="papers.db",
        pdf_dir="./arxiv_pdfs"  # 指定PDF保存目录
    )