from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA
# load agents and tools modules
import pandas as pd
from io import StringIO
from langchain.tools.python.tool import PythonAstREPLTool
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain import LLMMathChain

import pinecone
import chainlit as cl

@cl.langchain_factory(use_async=False)
async def init():

    # initialize pinecone client and connect to pinecone index
    pinecone.init(
        api_key="275bf42f-7112-4c33-94b7-7e821c83bcc4",  
        environment="us-west4-gcp-free"  
    ) 

    index_name = 'tk-policy'
    index = pinecone.Index(index_name) # connect to pinecone index


    # initialize embeddings object; for use with user query/input
    embed = OpenAIEmbeddings(model="text-embedding-ada-002")

    # initialize langchain vectorstore(pinecone) object
    text_field = 'text' # key of dict that stores the text metadata in the index
    vectorstore = Pinecone(
        index, embed.embed_query, text_field
    )

    # initialize LLM object
    llm = AzureChatOpenAI(deployment_name="gpt-35-turbo", temperature=0)

    # initialize vectorstore retriever object
    timekeeping_policy = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
    )

    # create employee data tool 
    with open("employee_data.csv", 'r') as f:
        # Read the file
        contents = f.read()

    csv_file = StringIO(contents) 
    df = pd.read_csv(csv_file) # load employee_data.csv as dataframe
    python = PythonAstREPLTool(locals={"df": df}) # set access of python_repl tool to the dataframe

    # create calculator tool
    calculator = LLMMathChain.from_llm(llm=llm, verbose=True)

    # create variables for f strings embedded in the prompts
    user = 'Alexander Verdad' # set user
    df_columns = df.columns.to_list() # print column names of df

    # prep the (tk policy) vectordb retriever, the python_repl(with df access) and langchain calculator as tools for the agent
    tools = [
        Tool(
            name = "Timekeeping Policies",
            func=timekeeping_policy.run,
            description="""
            Useful for when you need to answer questions about employee timekeeping policies.
            """
        ),
        Tool(
            name = "Employee Data",
            func=python.run,
            description = f"""
            Useful for when you need to answer questions about employee data stored in pandas dataframe 'df'. 
            Run python pandas operations on 'df' to help you get the right answer.
            'df' has the following columns: {df_columns}
            
            <user>: How many Sick Leave do I have left?
            <assistant>: df[df['name'] == '{user}']['sick_leave']
            <assistant>: You have n sick leaves left.              
            """
        ),
        Tool(
            name = "Calculator",
            func=calculator.run,
            description = f"""
            Useful when you need to do math operations or arithmetic.
            """
        )
    ]

    # change the value of the prefix argument in the initialize_agent function. This will overwrite the default prompt template of the zero shot agent type
    agent_kwargs = {'prefix': f'You are friendly HR assistant. You are tasked to assist the current user: {user} on questions related to HR. You have access to the following tools:'}


    # initialize the LLM agent
    return initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, agent_kwargs=agent_kwargs)

