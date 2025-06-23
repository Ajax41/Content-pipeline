from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Summary prompt
summary_prompt = PromptTemplate(
    input_variables=["blog_content"],
    template="""
You are a summarization assistant.

Summarize the following blog post into 3–5 key bullet points.
Make the summary clear, concise, and valuable for busy readers.

Blog Post:
{blog_content}

TL;DR Summary:
- 
"""
)

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.5)

# Create the summary chain
summary_chain = LLMChain(
    llm=llm,
    prompt=summary_prompt,
    verbose=True
)

# Function to run the summary agent
def run_summary_agent(blog_content: str):
    summary = summary_chain.run(blog_content=blog_content)
    return summary

# Run with example input
if __name__ == "__main__":
    blog = """ Title: Leveraging AI, NLP, and Predictive Analytics for Unprecedented Content Marketing Success

Introduction: The Future of Content Marketing 

In the ever-evolving landscape of content marketing, it's vital to stay ahead of the game. Current trends and advancements have paved the way for innovative approaches that remove guesswork from the equation, replacing it with data-driven strategies. One such game-changer is Artificial Intelligence (AI). Alongside Natural Language Processing (NLP) and Predictive Analytics, AI is revolutionizing the world of content marketing, boosting engagement, relevance, and return on investment (ROI) to unprecedented levels. 

Body

The Power of AI in Personalizing Content at Scale

AI is not just a buzzword. It's a potent tool that is reshaping the way we create and distribute content. AI's ability to personalize content at scale has made it an indispensable asset in the marketing toolkit. It can analyze vast data sets, learn from user behavior, and subsequently tailor content to individual preferences.

This high level of personalization has proven to dramatically improve engagement. In fact, studies have shown that AI can boost engagement rates by a whopping 45%. This is because consumers appreciate content that caters to their unique interests and needs. By leveraging AI, marketers can ensure that their content resonates with the audience, thereby fostering stronger connections and driving conversions.

NLP: Enhancing Relevance and Content Effectiveness

Natural Language Processing (NLP), a subset of AI, focuses on the interaction between computers and humans. In content marketing, NLP shines by boosting the relevance of your content, thereby increasing its effectiveness by 20%. 

NLP can understand, interpret, and generate human language in a meaningful way. It allows marketers to comprehend the sentiment behind user behaviors and tailor their content accordingly. This results in relevant and highly targeted content that truly speaks to the audience, ultimately leading to increased engagement and conversions.

Predictive Analytics: Maximizing Content ROI

The magic of predictive analytics lies in its ability to leverage historical data to predict future outcomes. In the realm of content marketing, this means utilizing past performance metrics to forecast the potential success of future content.

Latest studies reveal that predictive analytics can lead to a 25% higher content ROI. It allows marketers to understand what type of content resonates best with their audience, which platforms yield the most engagement, and what times are most effective for publishing. By arming themselves with this knowledge, marketers can make informed decisions, optimize their strategies, and maximize their content ROI.

Conclusion: The Dawn of a New Era in Content Marketing

As the digital landscape continues to evolve, AI, NLP, and predictive analytics are becoming increasingly critical for businesses to stay competitive. Embracing these technologies can significantly improve engagement, content effectiveness, and ROI, leading to a more successful content marketing strategy.

It's time to step into the future of content marketing, where data-driven decisions reign supreme, personalization is the norm, and predicting success is no longer a shot in the dark. Welcome to the era of AI, NLP, and predictive analytics – a revolutionized approach to content marketing that promises unprecedented growth and success.
"""
    summary = run_summary_agent(blog)
    print("TL;DR:\n", summary)
