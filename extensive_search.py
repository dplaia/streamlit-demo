import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import controlflow as cf

from agent_utils import *
from agent_tools import *
import argparse

import argparse
import os
import textwrap
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Import necessary libraries for new formats like csv, html, pdf, etc.
import pandas as pd
from markdown2 import markdown
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from reportlab.lib.units import inch

reasoningAgentChat = ReasoningModel()
basicSearchAgent = BasicSearchModel()

def ask_reasoning_model(content: str) -> str:
    global reasoningAgentChat

    response = reasoningAgentChat(content)
    return response

def google_single_search(search_query: str) -> str:
    global basicSearchAgent
    result_text = basicSearchAgent(search_query)
    return result_text

def google_search(search_queries: list[str]) -> list[str]:
    results = []
    for query in search_queries:
        print(f"Search query: {query}")
        response = google_single_search(query)
        results.append(response)
    return results

def search_query_help(search_query: str) -> str:
    query = f"""
    We have to improve the search results given a user search query. 
    Please think of multiple google search queries (plain text) that would increase the quality of the search results, when we combine all the search results.

    # Desired output format:

    Google Search Queries:
    - Search query 1
    - Search query 2
    - etc.

    Please sort the queries based on relevance/impact. First queries (Search query 1,2,3) are more relevant than last.

    # User Search Query:
    {search_query}

    """
    
    text_response = reasoningAgentChat(query)
    return text_response

class SearchQueryAgentResponse(BaseModel):
    google_search_queries: list[str] = Field(description="The extracted google search queries (if available).")
    # google_scholar_queries: list[str] | None = Field(description="The extracted google scholar queries (if available).")
    # text_summary: str  | None = Field(description="Extract the text summary here (if available).")

searchQueryAgent = cf.Agent(
    name="Query",
    model = "google/gemini-2.0-flash",
    instructions=f"""
    Your goal is to extract search queries (e.g., google search, google scholar, etc.) that are mention in a text.
    """
)

def generate_research_report(user_search_query, result_text) -> str:
    global reasoningAgentChat

    response = reasoningAgentChat(f"""
        Please generate a high quality report text based on the search results.
        
        The output should be in Markdown format, but please don't wrap the text like this:
        ```markdown
        # The report
        ```

        Add the correct citations in the writen report in the following style:
        "... outage caused disruption to online gaming and affected the stock market [1]. "

        Add a reference section with citations at the end. Use the urls that are present in the input text, don't use other urls.

        Format:

        # References

        [1] [thehindu.com](https://...) # always add two spaces at the end -> "  "
        [2] [google.com](https://...)
        [3] [indianexpress.com](https://...)
        [4] [hindustantimes.com](https://...)
        [5] [aljazeera.com](https://...)
        [6] etc. 

        The text should be as relevant to the user search query as possible. The lenght of the report should depend on the amound of sources available to you and based on the user query. 
        More sources with more diverse information (and more user questions) means longer report.
        The language report depends on the user query. If the user query is in English, write the report only in English. If the user query is in German, write the report only in German, etc. 

        # User Search Query:
        {user_search_query} 

        #Search Results (text):
        {result_text}


        """)    

    return response

def get_search_result_text(results: list[str]) -> str:
    result_text = ""
    for k, result in enumerate(results):
        #cprint(result)
        result_text += f"""
        # Result query {k+1}:

        {result}

        """
    return result_text

def run_research(search_query: str, max_searches: int = 10) -> str:
    search_queries_text = search_query_help(search_query)
    cprint(search_queries_text)

    task_extract_search_queries = cf.Task(
        objective=f"Extract all search queries from the input text",
        instructions=f"Extract the google search and google scholar search queries from the text. Here is the input text:  {search_queries_text}",
        result_type=SearchQueryAgentResponse,
        agents=[searchQueryAgent]
    )

    result = task_extract_search_queries.run()
    results = google_search(result.google_search_queries[:max_searches])
    result_text = get_search_result_text(results)

    print("Writing report now ...")
    response = generate_research_report(search_query, result_text)
    
    return response


def save_in_markdown(content, filename="output.md"):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)

def save_in_csv(content, filename="output.csv"):
    # Assuming `content` is a list of dictionaries or structured data
    df = pd.DataFrame(content)
    df.to_csv(filename, index=False)

def save_in_excel(content, filename="output.xlsx"):
    df = pd.DataFrame(content)
    df.to_excel(filename, index=False)

def save_in_pdf(content, filename="output.pdf"):
    # Create the PDF document with proper margins
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        rightMargin=72,  # 1 inch margins
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    # Create a list to hold the content
    story = []
    
    # Create a custom paragraph style
    styles = getSampleStyleSheet()
    custom_style = ParagraphStyle(
        'CustomStyle',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,  # Line spacing
        spaceAfter=10,  # Space between paragraphs
        wordWrap='CJK'  # Ensures proper word wrapping
    )
    
    # Split content into paragraphs and create Paragraph objects
    paragraphs = content.split('\n')
    for text in paragraphs:
        if text.strip():  # Only process non-empty lines
            p = Paragraph(text, custom_style)
            story.append(p)
            story.append(Spacer(1, 2))  # Add small space between paragraphs
    
    # Build the PDF
    doc.build(story)


def save_in_html(content, filename="output.html"):
    html_content = markdown(content)  # Convert Markdown to HTML
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)


def main():
    parser = argparse.ArgumentParser(description="Run extended web searches.")
    parser.add_argument("query", type=str, help="The initial search query.")
    parser.add_argument("-m", "--max_searches", type=int, default=5, help="Maximum number of searches to perform.")
    parser.add_argument("-f", "--format", type=str, choices=['markdown', 'csv', 'excel', 'pdf', 'html'], default='markdown', help="Choose output format.")

    args = parser.parse_args()

    # Get the report text from the research process
    report_text = run_research(args.query, args.max_searches)

    # Save output in chosen format
    if args.format == 'markdown':
        save_in_markdown(report_text)  # Save the actual report content
    elif args.format == 'csv':
        save_in_csv([{'result': report_text}])  # Wrap content in dict for CSV
    elif args.format == 'excel':
        save_in_excel([{'result': report_text}])  # Wrap content in dict for Excel
    elif args.format == 'pdf':
        save_in_pdf(report_text)
    elif args.format == 'html':
        save_in_html(report_text)

    print(f"Output saved as {args.format}")
'''
    if args.save:
        with open("markdown_output.md", "w") as f:
            f.write(report_text)

    cprint(report_text)
'''

if __name__ == "__main__":
    main()