from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# ✅ Load OpenAI API key from environment
openai_api_key = os.getenv("OPENAI_API_KEY")

# ✅ Define your LLM
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.7,
    api_key=openai_api_key
)

# 1. Research Agent
research_prompt = PromptTemplate(
    input_variables=["topic"],
    template="""
You are a research assistant. Your job is to gather key insights on the topic: {topic}.
Return the research as 3–5 key points. Include statistics, trends, or noteworthy facts.

Respond in this format:
- Point 1
- Point 2
- Point 3
- (Optional) Point 4/5
"""
)
research_chain = LLMChain(llm=llm, prompt=research_prompt, output_key="research")

# 2. Writing Agent
writing_prompt = PromptTemplate(
    input_variables=["research"],
    template="""
You are a skilled content writer.

Turn the following research bullet points into a compelling blog article. The article should be clear, informative, and engaging for a marketing professionals audience. Use a professional tone.

Research:
{research}

Write the blog post with an introduction, body (with headers), and conclusion.
"""
)
writing_chain = LLMChain(llm=llm, prompt=writing_prompt, output_key="blog")

# 3. Summary Agent
summary_prompt = PromptTemplate(
    input_variables=["blog"],
    template="""
You are a content strategist.

Summarize the following blog post into a professional TL;DR bullet list for internal marketing reports.

Blog:
{blog}

Respond in this format:
- Point 1
- Point 2
- Point 3
- (Optional) Point 4/5
"""
)
summary_chain = LLMChain(llm=llm, prompt=summary_prompt, output_key="summary")

# 4. Social Media Agent
social_prompt = PromptTemplate(
    input_variables=["summary"],
    template="""
You are a social media content strategist.

Based on the following summary of a blog post, generate:
1. A professional LinkedIn post
2. A 4-part Twitter thread
3. An engaging Instagram caption

Summary:
{summary}

Respond in this format:

LinkedIn Post:
...

Twitter Thread:
1.
2.
3.
4.

Instagram Caption:
...
"""
)
social_chain = LLMChain(llm=llm, prompt=social_prompt, output_key="social")

# ✅ Create and expose full pipeline
pipeline = SequentialChain(
    chains=[research_chain, writing_chain, summary_chain, social_chain],
    input_variables=["topic"],
    output_variables=["research", "blog", "summary", "social"],
    verbose=True
)
