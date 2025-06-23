from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Social content prompt
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

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.7)

# Create the social content chain
social_chain = LLMChain(
    llm=llm,
    prompt=social_prompt,
    verbose=True
)

# Function to run the social agent
def run_social_agent(summary: str):
    content = social_chain.run(summary=summary)
    return content

# Run with example input
if __name__ == "__main__":
    summary = """ - Artificial Intelligence (AI), Natural Language Processing (NLP), and Predictive Analytics are revolutionizing content marketing by enabling data-driven strategies, boosting engagement, relevance, and return on investment (ROI).
- AI personalizes content at scale by analyzing data and learning from user behavior, which can increase engagement rates by up to 45%.
- NLP, a subset of AI, enhances the relevance and effectiveness of content by understanding, interpreting, and generating human language, leading to a 20% increase in content effectiveness.
- Predictive analytics uses historical data to forecast the potential success of future content, leading to a 25% higher content ROI by helping marketers understand what content, platforms, and times are most effective.
- As the digital landscape evolves, embracing AI, NLP, and predictive analytics is crucial for businesses to stay competitive, improve engagement, content effectiveness, and ROI, leading to a more successful content marketing strategy."""
    content = run_social_agent(summary)
    print("Social Content:\n", content)
