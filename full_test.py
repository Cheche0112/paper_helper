from arxiv_crawler import ArxivScraper
import asyncio
import os
from datetime import date

async def main():
    try:
        # 设置日期范围（2024年2月1日到2月5日，缩小范围以加快测试）
        date_from = "2024-02-01"
        date_until = "2024-02-05"
        
        print(f"开始爬取从 {date_from} 到 {date_until} 的论文...")
        
        # 创建输出目录
        os.makedirs("./output_test", exist_ok=True)
        
        # 创建爬虫实例
        scraper = ArxivScraper(
            date_from=date_from,
            date_until=date_until,
            trans_to=None,  # 暂时禁用翻译
            optional_keywords=[
                "LLM", "GPT", "AI"
            ],  # 设置关键词
            category_whitelist=[
                "cs.AI", "cs.CL", "cs.LG"
            ],  # 设置分类白名单
            proxy=None  # 暂时不使用代理
        )
        
        # 运行爬虫（只获取前100篇论文以加快测试）
        print("开始获取论文...")
        await scraper.fetch_all()
        print(f"爬取完成！共获取 {len(scraper.papers)} 篇论文")
        
        # 处理论文
        print("处理论文...")
        scraper.process_papers()
        
        # 导出为Markdown
        print("导出为Markdown...")
        scraper.paper_exporter.to_markdown(
            output_dir="./output_test", 
            filename_format="%Y-%m-%d",
            metadata=scraper.meta_data
        )
        print("结果已保存到 output_test 目录")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # 确保关闭会话
        if hasattr(scraper, 'session') and scraper.session:
            await scraper.close_session()

if __name__ == "__main__":
    asyncio.run(main()) 