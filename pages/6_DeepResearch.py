import streamlit as st
import numpy as np

from agent_tools import BasicSearchModel, ReasoningModel
from extensive_search import run_research
from write_article import get_article

st.set_page_config(page_title="DeepResearchHS", 
                    page_icon=":books:", 
                    layout="wide", 
                    initial_sidebar_state="expanded",
                    menu_items={
                        'About': """
                        # DeepResearch GUI
                        
                        """,
                        'Get Help': 'https://www.streamlit.io/'
                    })

def main():
    st.title("DeepResearchHS")

    # create random table fo numbers
    #data = np.random.rand(10, 10)
    #st.write(data)
    search_query_text = st.text_area("Search Query:", height=180)
    
    # add combobox for model selection
    model_selector = st.selectbox("Mode:", ["Quick Search", "Get Article", "DeepResearch"])
    button = st.button("Start DeepSearch")


    if button:
        response = ""
        response_title = ""

        if model_selector == "Quick Search":
            searchAgent = BasicSearchModel() 
            response = searchAgent(search_query_text)
            response_title = "Quick Search Result"

        elif model_selector == "Get Article":
            response = get_article(search_query_text)
            response_title = "Article Result"
        elif model_selector == "DeepResearch":
            response = run_research(search_query_text)
            response_title = "DeepResearch Result"

        st.markdown(f"""
        {response}

        """)

if __name__ == "__main__":
    main()