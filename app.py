import streamlit as st
from content_pipeline import pipeline
import os

st.set_page_config(page_title="AI Content Pipeline", layout="wide")

st.title("🚀 AI-Powered Content Pipeline")
st.write("Generate research, a blog post, summary, and social media content from a single topic.")

topic = st.text_input("Enter a content topic:")

uploaded_file = st.file_uploader("Optional: Upload a file to enrich the topic", type=["txt", "md"])
file_contents = ""
if uploaded_file is not None:
    file_contents = uploaded_file.read().decode("utf-8")

if st.button("Generate Content") and (topic or file_contents):
    with st.spinner("Generating content..."):
        full_topic = topic
        if file_contents:
            full_topic += f"\n\nAdditional Notes:\n{file_contents}"
        outputs = pipeline.invoke({"topic": full_topic})

    st.success("✅ Content generated successfully!")

    def display_section(title, content, key):
        with st.expander(title, expanded=True):
            st.markdown(f"```\n{content}\n```")

    display_section("🔍 Research", outputs["research"], "research")
    display_section("✍️ Blog Post", outputs["blog"], "blog")
    display_section("📝 Summary", outputs["summary"], "summary")
    display_section("📣 Social Media Content", outputs["social"], "social")
