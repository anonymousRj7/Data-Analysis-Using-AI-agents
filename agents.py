import pandas as pd
from pandasai import SmartDataframe
from langchain_groq.chat_models import ChatGroq
from dotenv import load_dotenv
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import os



load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

def get_llm(model_name):
    
    llm = ChatGroq(
        model_name=model_name,
        groq_api_key = api_key

    )

    return llm


def smartdataframe_respone(data,Query,llm):
    df = SmartDataframe(data, config={"llm": llm})

    prompt = Query
    return df.chat(prompt)

def response_agent(data,Query,llm):

    response_agent = create_pandas_dataframe_agent(
        llm,
        data,
        verbose=True,
        handle_parsing_errors=True
        )
    
    prompt = Query

    respone = response_agent.invoke(prompt)["output"]

    return respone

def code_agent(data,Query,llm):

    code_agent = create_pandas_dataframe_agent(
        llm,
        data,
        verbose=True,
        handle_parsing_errors=True,
        )
    
    prompt =  "Please provide a well-formatted and easy-to-understand Python code snippet for the  query : "  + Query + "  : from the  dataframe.And Don't provide anything else.After give the code finished the chian."


#     prompt =  f"""Given Query: {Query}
# # Please provide a well-formatted and easy-to-understand Python code snippet that [state the purpose or functionality of the code you're requesting]. The code should follow best coding practices, including:

# # - Proper indentation and formatting
# # - Relevant comments explaining the code logic and functionality
# # - Meaningful variable and function names
# # - Error handling and input validation (if applicable)
# # - Modular structure (e.g., separating code into functions/classes)
# # - Appropriate use of data structures and algorithms
# # - Efficient and optimized code (if applicable)

# # If possible, please also provide a brief explanation of how the code works and any assumptions or dependencies it might have."""

    respone = code_agent.invoke(prompt)["output"]

    return respone

def graph_agent(data,Query,llm):

    graph_agent = create_pandas_dataframe_agent(
        llm,
        data,
        verbose=True,
        handle_parsing_errors=True,
        )
    
    prompt =  "If this query : "  + Query + "  : contain some ploting of graph of then show the graph else finished the chain "

    respone = graph_agent.invoke(prompt)["output"]

    return respone