from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4", temperature=0.7)

# Prompts
research_prompt = PromptTemplate.from_template(
    "You are a research assistant. Provide 3–5 bullet points on: {topic}"
)
writing_prompt = PromptTemplate.from_template(
    "Write a blog post based on:\n{research}"
)
summary_prompt = PromptTemplate.from_template(
    "Summarize this blog post:\n{blog}"
)
social_prompt = PromptTemplate.from_template(
    "Write a LinkedIn post, Twitter thread, and IG caption based on:\n{summary}"
)

# Chains
pipeline = SequentialChain(
    chains=[
        LLMChain(llm=llm, prompt=research_prompt, output_key="research"),
        LLMChain(llm=llm, prompt=writing_prompt, output_key="blog"),
        LLMChain(llm=llm, prompt=summary_prompt, output_key="summary"),
        LLMChain(llm=llm, prompt=social_prompt, output_key="social"),
    ],
    input_variables=["topic"],
    output_variables=["research", "blog", "summary", "social"],
    verbose=True
)

