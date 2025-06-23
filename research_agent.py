from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

# Load API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize model
llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.7, model="gpt-3.5-turbo")

# Define prompt
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

# Create the research chain
research_chain = LLMChain(
    llm=llm,
    prompt=research_prompt,
    verbose=True
)

# Function to run the agent
def run_research_agent(topic: str):
    result = research_chain.run(topic=topic)
    return result

# Run it with a sample topic
if __name__ == "__main__":
    topic = "Which colors or color categories are trending in fashion and design for 2025"
    output = run_research_agent(topic)
    print("Research Output:\n", output)




