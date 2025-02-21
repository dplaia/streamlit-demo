import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import controlflow as cf

from agent_utils import *
from agent_tools import *
import argparse


FLASH2_MODEL = "google/" + config.FLASH2_MODEL

#reasoningModel = ReasoningModel()

def use_reasoning(content: str) -> str:
    """
    Use this function to reason about anything.

    Input:
        content (str): The content that you want to reason about.

    Output:
        str: The reasoned response/output.
    """
    newModel = ReasoningModel()
    return newModel(content)

def get_article(search_query: str) -> str:
    text = f"""
    We need help with writing a two page article about the following topic.
    Please help me with this article. Provide me with some conent and methods to achieve this article.

    # Here is the question/topic:
    {search_query}
    """

    reasoning_response = use_reasoning(text)

    task = cf.Task(f"Write an article based on a given topic. Use the following information to help you with the text: {reasoning_response}")

    article_output = task.run()

    return article_output

def main():

    parser = argparse.ArgumentParser(description="Run extended web searches.")
    parser.add_argument("query", type=str, help="The initial search query.")
    parser.add_argument("-s", "--save", action="store_true", help="Save the report as markdown file.")

    args = parser.parse_args()

    text = f"""
    We need help with writing a two page article about the following topic.
    Please help me with this article. Provide me with some conent and methods to achieve this article.

    # Here is the question/topic:
    {args.query}
    """

    reasoning_response = use_reasoning(text)

    task = cf.Task(f"Write an article based on a given topic. Use the following information to help you with the text: {reasoning_response}")

    article_output = task.run()

    console_print(article_output)

    if args.save:
        with open("article_output.md", "w") as f:
            f.write(article_output)

    console_print(article_output)


if __name__ == "__main__":
    main()