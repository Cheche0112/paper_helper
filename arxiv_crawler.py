import argparse
import asyncio
from datetime import date
import asyncio
import re
from datetime import datetime, timedelta, UTC
from itertools import chain
import ssl
import logging
import random

import aiohttp
from aiohttp import ClientTimeout
from bs4 import BeautifulSoup, NavigableString, Tag
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from arxiv_time import next_arxiv_update_day
from paper import Paper, PaperDatabase, PaperExporter


class ArxivScraper(object):
    def __init__(
        self,
        date_from,
        date_until,
        category_blacklist=[],
        category_whitelist=["cs.CV", "cs.AI", "cs.LG", "cs.CL", "cs.IR", "cs.MA"],
        optional_keywords=["personality" ,"LLM", "LLMs", "language model", "persona"],
        trans_to="zh-CN",
        proxy=None,
    ):
        """
        一个抓取指定日期范围内的arxiv文章的类,
        搜索基于https://arxiv.org/search/advanced,
        一个文件被爬取到的条件是：首次提交时间在`date_from`和`date_until`之间，并且包含至少一个关键词。
        一个文章被详细展示（不被过滤）的条件是：至少有一个领域在白名单中，并且没有任何一个领域在黑名单中。
        翻译基于google-translate

        Args:
            date_from (str): 开始日期(含当天)
            date_until (str): 结束日期(含当天)
            category_blacklist (list, optional): 黑名单. Defaults to [].
            category_whitelist (list, optional): 白名单. Defaults to ["cs.CV", "cs.AI", "cs.LG", "cs.CL", "cs.IR", "cs.MA"].
            optional_keywords (list, optional): 关键词, 各词之间关系为OR, 在标题/摘要中至少要出现一个关键词才会被爬取.
                Defaults to [ "LLM", "LLMs", "language model", "language models", "multimodal", "finetuning", "GPT"]
            trans_to: 翻译的目标语言, 若设为可转换为False的值则不会翻译
            proxy (str | None, optional): 用于翻译和爬取arxiv时要使用的代理, 通常是http://127.0.0.1:7890. Defaults to None
        """
        # announced_date_first 日期处理为年月，从from到until的所有月份都会被爬取
        # 如果from和until是同一个月，则until设置为下个月(from+31)
        self.search_from_date = datetime.strptime(date_from[:-3], "%Y-%m")
        self.search_until_date = datetime.strptime(date_until[:-3], "%Y-%m")
        if self.search_from_date.month == self.search_until_date.month:
            self.search_until_date = (self.search_from_date + timedelta(days=31)).replace(day=1)
        # 由于arxiv的奇怪机制，每个月的第一天公布的文章总会被视作上个月的文章, 所以需要将月初文章的首次公布日期往后推一天
        self.fisrt_announced_date = next_arxiv_update_day(next_arxiv_update_day(self.search_from_date) + timedelta(days=1))

        self.category_blacklist = category_blacklist  # used as metadata
        self.category_whitelist = category_whitelist  # used as metadata
        self.optional_keywords = [kw.replace(" ", "+") for kw in optional_keywords]  # url转义

        self.trans_to = trans_to  # translate
        self.proxy = proxy

        self.filt_date_by = "announced_date_first"  # url
        self.order = "-announced_date_first"  # url(结果默认按首次公布日期的降序排列，这样最新公布的会在前面)
        self.total = None  # fetch_all
        self.step = 50  # url, fetch_all
        self.papers: list[Paper] = []  # fetch_all

        self.paper_db = PaperDatabase()
        self.paper_exporter = PaperExporter(date_from, date_until, category_blacklist, category_whitelist)
        self.console = Console()
        self.session = None

    @property
    def meta_data(self):
        """
        返回搜索的元数据
        """
        return dict(repo_url="https://github.com/huiyeruzhou/arxiv_crawler", **self.__dict__)

    def get_url(self, start):
        """
        获取用于搜索的url

        Args:
            start (int): 返回结果的起始序号, 每个页面只会包含序号为[start, start+50)的文章
            filter_date_by (str, optional): 日期筛选方式. Defaults to "submitted_date_first".
        """
        # https://arxiv.org/search/advanced?terms-0-operator=AND&terms-0-term=LLM&terms-0-field=all&terms-1-operator=OR&terms-1-term=language+model&terms-1-field=all&terms-2-operator=OR&terms-2-term=multimodal&terms-2-field=all&terms-3-operator=OR&terms-3-term=finetuning&terms-3-field=all&terms-4-operator=AND&terms-4-term=GPT&terms-4-field=all&classification-computer_science=y&classification-physics_archives=all&classification-include_cross_list=include&date-year=&date-filter_by=date_range&date-from_date=2024-08-08&date-to_date=2024-08-15&date-date_type=submitted_date_first&abstracts=show&size=50&order=submitted_date
        kwargs = "".join(
            f"&terms-{i}-operator=OR&terms-{i}-term={kw}&terms-{i}-field=all"
            for i, kw in enumerate(self.optional_keywords)
        )
        date_from = self.search_from_date.strftime("%Y-%m")
        date_until = self.search_until_date.strftime("%Y-%m")
        return (
            f"https://arxiv.org/search/advanced?advanced={kwargs}"
            f"&classification-computer_science=y&classification-physics_archives=all&"
            f"classification-include_cross_list=include&"
            f"date-year=&date-filter_by=date_range&date-from_date={date_from}&date-to_date={date_until}&"
            f"date-date_type={self.filt_date_by}&abstracts=show&size={self.step}&order={self.order}&start={start}"
        )

    async def create_session(self):
        """创建会话"""
        if self.session is None:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout, connector=connector, headers=headers)
        return self.session
        
    async def close_session(self):
        """关闭会话"""
        if self.session is not None:
            await self.session.close()
            self.session = None

    async def fetch_content(self, url):
        """获取网页内容"""
        session = await self.create_session()
        
        for attempt in range(3):
            try:
                # 添加随机延迟，避免请求过快
                await asyncio.sleep(random.uniform(1, 3))
                
                async with session.get(url, proxy=self.proxy) as response:
                    if response.status == 200:
                        content = await response.text()
                        if content:
                            return content
                        else:
                            logging.error(f"Empty content received from {url}")
                    elif response.status == 429:  # Too Many Requests
                        logging.error(f"Rate limited (429). Waiting before retry...")
                        await asyncio.sleep(10 * (attempt + 1))  # 增加等待时间
                    else:
                        logging.error(f"HTTP {response.status} error for {url}")
                        
            except aiohttp.ClientError as e:
                logging.error(f"Network error: {str(e)}")
                if attempt < 2:
                    await asyncio.sleep(5 * (attempt + 1))
                continue
            except Exception as e:
                logging.error(f"Unexpected error: {str(e)}")
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                continue
            
        logging.error(f"Failed to fetch content after 3 attempts: {url}")
        return None

    async def fetch_all(self):
        """获取所有论文"""
        try:
            self.console.log(f"[bold green]Fetching the first {self.step} papers...")
            self.console.print(f"[grey] {self.get_url(0)}")
            
            # 第一次请求
            content = await self.fetch_content(self.get_url(0))
            if content is None:
                self.console.log("[bold red]Failed to fetch initial content")
                return
                
            # 解析总数
            self.total = self.parse_total(content)
            if self.total is None:
                self.console.log("[bold red]Failed to parse total number of papers")
                return
                
            self.console.log(f"Found {self.total} papers in total")
            initial_papers = self.parse_search_html(content)
            if initial_papers:
                self.papers.extend(initial_papers)
            
            # 创建任务列表，每批次最多10个并发请求
            batch_size = 10
            with Progress() as progress:
                task = progress.add_task("Fetching", total=self.total)
                
                for start in range(self.step, self.total, self.step * batch_size):
                    batch_tasks = []
                    for offset in range(0, min(self.step * batch_size, self.total - start), self.step):
                        current_start = start + offset
                        
                        async def wrapper(current_start):
                            content = await self.fetch_content(self.get_url(current_start))
                            if content is None:
                                self.console.log(f"[bold red]Failed to fetch content for start={current_start}")
                                return None
                            return self.parse_search_html(content)
                            
                        batch_tasks.append(wrapper(current_start))
                    
                    # 等待当前批次完成
                    batch_results = await asyncio.gather(*batch_tasks)
                    for papers in batch_results:
                        if papers is not None:
                            self.papers.extend(papers)
                    
                    # 更新进度
                    progress.update(task, completed=min(start + self.step * batch_size, self.total))
                    
                    # 批次之间添加短暂延迟
                    await asyncio.sleep(2)
                    
            self.console.log("Fetching completed.")
            
        except Exception as e:
            self.console.log(f"[bold red]Error during fetch_all: {str(e)}")
            raise
            
        finally:
            await self.close_session()

    def fetch_update(self):
        """
        更新文章, 这会从最新公布的文章开始更新, 直到遇到已经爬取过的文章为止。
        为了效率，建议在运行fetch_all后再运行fetch_update
        """
        # 当前时间
        utc_now = datetime.now(UTC).replace(tzinfo=None)
        # 上一次更新最新文章的UTC时间. 除了更新新文章外也可能重新爬取了老文章, 数据库只看最新文章的时间戳。
        last_update = self.paper_db.newest_update_time()
        # 检查一下上次之后的最近一个arxiv更新日期
        self.search_from_date = next_arxiv_update_day(last_update)
        self.console.log(f"[bold yellow]last update: {last_update.strftime('%Y-%m-%d %H:%M:%S')}, "
                         f"next arxiv update: {self.search_from_date.strftime('%Y-%m-%d')}" 
                         )
        self.console.log(f"[bold yellow]UTC now: {utc_now.strftime('%Y-%m-%d %H:%M:%S')}")
        # 如果还没到更新时间就不更新了
        if self.search_from_date >= utc_now:
            self.console.log(f"[bold red]Your database is already up to date.")
            return
        # 如果这一次的更新时间恰好是这个月的第一个更新日，那么当日更新的文章都会出现在上个月的搜索结果中
        # 为了正确获得这天的文章，我们上推一个月的搜索时间
        self.fisrt_announced_date = self.search_from_date
        if self.search_from_date == next_arxiv_update_day(self.search_from_date.replace(day=1)):
            self.search_from_date = self.search_from_date - timedelta(days=31)
            self.console.log(f"[bold yellow]The update in {self.fisrt_announced_date.strftime('%Y-%m-%d')} can only be found in the previous month.")
        else:
            self.console.log(
                f"[bold green]Searching from {self.search_from_date.strftime('%Y-%m-%d')} "
                f"to {self.search_until_date.strftime('%Y-%m-%d')}, fetch the first {self.step} papers..."
            )
        self.console.print(f"[grey] {self.get_url(0)}")

        continue_update = self.update(0)
        for start in range(self.step, self.total, self.step):
            if not continue_update:
                break

            continue_update = self.update(start)
        self.console.log(f"[bold green]Fetching completed. {len(self.papers)} new papers.")
        # if self.trans_to:
        #     asyncio.run(self.translate())
        self.process_papers()

    def process_papers(self):
        """
        推断文章的首次公布日期, 并将文章添加到数据库中
        """
        # 从下一个可能的公布日期开始
        announced_date = next_arxiv_update_day(self.fisrt_announced_date)   
        self.console.log(f"fisrt announced date: {announced_date.strftime('%Y-%m-%d')}")
        # 按照从前到后的时间顺序梳理文章
        for paper in reversed(self.papers):
            # 文章于T日美东时间14:00(T UTC+0 18:00)前提交，将于T日美东时间20:00(T+1 UTC+0 00:00)公布，T始终为工作日。
            # 因此可知美东 T日的文章至少在UTC+0 T+1日公布，如果超过14:00甚至会在UTC+0 T+2日公布
            next_possible_annouced_date = next_arxiv_update_day(paper.first_submitted_date + timedelta(days=1))
            if announced_date < next_possible_annouced_date:
                announced_date = next_possible_annouced_date
            paper.first_announced_date = announced_date
        self.paper_db.add_papers(self.papers)
    
    def reprocess_papers(self):
        """
        这会从数据库中获取所有文章, 并重新推断文章的首次公布日期，并打印调试信息
        """
        self.papers = self.paper_db.fetch_all()
        self.process_papers()
        with open("announced_date.csv", "w") as f:
            f.write("url,title,announced_date,submitted_date\n")
            for paper in self.papers:
                f.write(
                    f"{paper.url},{paper.title},{paper.first_announced_date.strftime('%Y-%m-%d')},{paper.first_submitted_date.strftime('%Y-%m-%d')}\n"
                )

    def update(self, start) -> bool:
        content = asyncio.run(self.fetch_content(self.get_url(start)))
        self.papers.extend(self.parse_search_html(content))
        cnt_new = self.paper_db.count_new_papers(self.papers[start : start + self.step])
        if cnt_new < self.step:
            self.papers = self.papers[: start + cnt_new]
            return False
        else:
            return True

    def parse_search_html(self, content) -> list[Paper]:
        """
        解析搜索结果页面, 并将结果保存到self.paper_result中
        初次调用时, 会解析self.total
        """
        if content is None:
            self.console.log("[bold red]Cannot parse None content")
            return []
        
        try:
            soup = BeautifulSoup(content, "html.parser")
            if not self.total:
                total_elements = soup.select("#main-container > div.level.is-marginless > div.level-left > h1")
                if not total_elements:
                    self.console.log("[bold red]Could not find total results element")
                    self.total = 0
                    return []
                    
                total = total_elements[0].text
                self.console.log(f"[grey]Found total text: {total}")
                
                # "Showing 1–50 of 2,542,002 results" or "Sorry, your query returned no results"
                if "Sorry" in total:
                    self.total = 0
                    return []
                    
                try:
                    total = total[total.find("of") + 3 : total.find("results")].replace(",", "").strip()
                    self.total = int(total)
                except (ValueError, IndexError) as e:
                    self.console.log(f"[bold red]Error parsing total: {str(e)}")
                    self.total = 0
                    return []

            results = soup.find_all("li", {"class": "arxiv-result"})
            papers = []
            for result in results:
                try:
                    url_tag = result.find("a")
                    url = url_tag["href"] if url_tag else "No link"

                    title_tag = result.find("p", class_="title")
                    title = self.parse_search_text(title_tag) if title_tag else "No title"
                    title = title.strip()

                    date_tag = result.find("p", class_="is-size-7")
                    date = date_tag.get_text(strip=True) if date_tag else "No date"
                    if "v1" in date:
                        v1 = date.find("v1submitted")
                        date = date[v1 + 12 : date.find(";", v1)]
                    else:
                        submit_date = date.find("Submitted")
                        date = date[submit_date + 9 : date.find(";", submit_date)]

                    category_tag = result.find_all("span", class_="tag")
                    categories = [
                        category.get_text(strip=True) 
                        for category in category_tag 
                        if "tooltip" in category.get("class", [])
                    ]

                    authors_tag = result.find("p", class_="authors")
                    authors = authors_tag.get_text(strip=True)[len("Authors:") :] if authors_tag else "No authors"

                    summary_tag = result.find("span", class_="abstract-full")
                    abstract = self.parse_search_text(summary_tag) if summary_tag else "No summary"
                    abstract = abstract.strip()

                    comments_tag = result.find("p", class_="comments")
                    comments = comments_tag.get_text(strip=True)[len("Comments:") :] if comments_tag else "No comments"

                    papers.append(
                        Paper(
                            url=url,
                            title=title,
                            first_submitted_date=datetime.strptime(date, "%d %B, %Y"),
                            categories=categories,
                            authors=authors,
                            abstract=abstract,
                            comments=comments,
                        )
                    )
                except Exception as e:
                    self.console.log(f"[bold red]Error parsing paper: {str(e)}")
                    continue
                    
            return papers
        except Exception as e:
            self.console.log(f"[bold red]Error in parse_search_html: {str(e)}")
            self.total = 0
            return []

    def parse_search_text(self, tag):
        string = ""
        for child in tag.children:
            if isinstance(child, NavigableString):
                string += re.sub(r"\s+", " ", child)
            elif isinstance(child, Tag):
                if child.name == "span" and "search-hit" in child.get("class"):
                    string += re.sub(r"\s+", " ", child.get_text(strip=False))
                elif child.name == "a" and ".style.display" in child.get("onclick"):
                    pass
                else:
                    import pdb

                    pdb.set_trace()
        return string

    async def translate(self):
        if not self.trans_to:
            raise ValueError("No target language specified.")
        self.console.log("[bold green]Translating...")
        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            console=self.console,
            transient=False,
        ) as p:
            total = len(self.papers)
            task = p.add_task(
                description=f"[bold green]Translating {total} papers",
                total=total,
            )

            async def worker(paper):
                await paper.translate(langto=self.trans_to)
                p.update(task, advance=1)

            await asyncio.gather(*[worker(paper) for paper in self.papers])

    def to_markdown(self, output_dir="./output_llms", filename_format="%Y-%m-%d", meta=False):
        self.paper_exporter.to_markdown(output_dir, filename_format, self.meta_data if meta else None)

    def to_csv(self, output_dir="./output_llms", filename_format="%Y-%m-%d",  header=False, csv_config={},):
        self.paper_exporter.to_csv(output_dir, filename_format, header, csv_config)

    def parse_total(self, content):
        """解析搜索结果总数"""
        try:
            # 使用BeautifulSoup解析HTML
            soup = BeautifulSoup(content, 'html.parser')
            
            # 查找包含结果数量的元素
            total_text = soup.find('h1', {'class': 'title'})
            if total_text:
                self.console.log(f"Found total text: {total_text.text.strip()}")
                # 提取数字
                match = re.search(r'Showing \d+–\d+ of ([\d,]+) results', total_text.text)
                if match:
                    total = int(match.group(1).replace(',', ''))
                    return total
                    
            self.console.log("[bold red]Could not find total results count in the page")
            return None
            
        except Exception as e:
            self.console.log(f"[bold red]Error parsing total: {str(e)}")
            return None


import argparse
import asyncio
from datetime import date, datetime

if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='ArxivScraper 参数配置')
    parser.add_argument('--month', required=True, type=str,
                        help='指定开始抓取的月份（格式: YYYY-MM），例如：2023-10')
    args = parser.parse_args()

    # 解析起始月份
    try:
        start_year, start_month = map(int, args.month.split('-'))
        # 验证月份有效性
        if not (1 <= start_month <= 12):
            raise ValueError("月份必须在1到12之间")
        start_date = date(start_year, start_month, 1)
    except ValueError:
        raise ValueError("无效的月份格式，请使用 YYYY-MM 格式（例如：2023-10）")

    # 获取当前年月
    current_date = date.today()
    current_year, current_month = current_date.year, current_date.month

    # 检查起始月份是否超过当前月份
    if (start_year > current_year) or (start_year == current_year and start_month > current_month):
        raise ValueError("起始月份不能晚于当前月份")

    # 初始化循环变量
    y, m = start_year, start_month

    while True:
        # 生成当前月份的起止日期
        date_from = date(y, m, 1)
        if m == 12:
            next_y, next_m = y + 1, 1
        else:
            next_y, next_m = y, m + 1
        date_until = date(next_y, next_m, 1)

        # 初始化并执行抓取器
        print(f"抓取月份: {y}-{m:02d}")
        scraper = ArxivScraper(
            date_from=date_from.strftime("%Y-%m-%d"),
            date_until=date_until.strftime("%Y-%m-%d"),
        )
        asyncio.run(scraper.fetch_all())

        # 到达当前月份则结束循环
        if y == current_year and m == current_month:
            break

        # 计算下一个月份
        m += 1
        if m > 12:
            m = 1
            y += 1