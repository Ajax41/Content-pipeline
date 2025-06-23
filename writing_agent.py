from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Writing prompt
writing_prompt = PromptTemplate(
    input_variables=["research_points", "tone", "audience"],
    template="""
You are a skilled content writer.

Turn the following research bullet points into a compelling blog article. The article should be clear, informative, and engaging for a {audience} audience. Use a {tone} tone.

Research:
{research_points}

Write the blog post with an introduction, body (with headers), and conclusion.
"""
)

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.7)

# Create the writing chain
writing_chain = LLMChain(
    llm=llm,
    prompt=writing_prompt,
    verbose=True
)

# Function to run the writing agent
def run_writing_agent(research_points, tone="professional", audience="marketing professionals"):
    response = writing_chain.run(
        research_points=research_points,
        tone=tone,
        audience=audience
    )
    return response

# Run with example input
if __name__ == "__main__":
    research_output = """
- AI personalizes content at scale, improving engagement by 45%.
- NLP boosts relevance, increasing content effectiveness by 20%.
- Predictive analytics lead to 25% higher content ROI.
"""
    article = run_writing_agent(research_output)
    print("Generated Blog Post:\n", article)
