from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
import chainlit as cl

template = """Question: {question}

Answer: Let's think step by step."""

@cl.langchain_factory(use_async=True)
def factory():
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=ChatOpenAI(temperature=0), verbose=True)

    return llm_chain