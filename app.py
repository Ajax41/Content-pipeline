import streamlit as st
st.write("✅ app.py is loading...")

from content_pipeline import pipeline

st.set_page_config(page_title="AI Content Pipeline", layout="wide")

st.title("🚀 AI-Powered Content Pipeline")
st.write("Generate research, a blog post, summary, and social media content from a single topic.")

# --- Topic Input ---
topic = st.text_input("Enter a content topic:", "")

# --- File Upload ---
uploaded_file = st.file_uploader("Optional: Upload a file to enrich the topic (e.g., notes or context)", type=["txt", "md"])
file_contents = ""
if uploaded_file is not None:
    file_contents = uploaded_file.read().decode("utf-8")

# --- Submit Button ---
if st.button("Generate Content") and (topic or file_contents):
    with st.spinner("Generating content..."):
        full_topic = topic
        if file_contents:
            full_topic += f"\n\nAdditional Notes:\n{file_contents}"

        outputs = pipeline.invoke({"topic": full_topic})

    st.success("✅ Content generated successfully!")

    # JavaScript for copy to clipboard functionality
    st.markdown("""
        <script>
        function copyToClipboard(text) {
            navigator.clipboard.writeText(text).then(function() {
                console.log('Copied to clipboard');
            }, function(err) {
                console.error('Could not copy text: ', err);
            });
        }
        </script>
    """, unsafe_allow_html=True)

    # Display function with styled copy icon
    def display_section(title, content, key):
        safe_key = key.replace(" ", "_")
        escaped_content = content.replace("`", "\\`").replace("\n", "\\n").replace('"', '\\"')
        with st.expander(f"{title}", expanded=True):
            st.markdown(
                f"""
                <div style='position: relative; padding-top: 10px;'>
                    <button onclick="copyToClipboard(`{escaped_content}`)" 
                            style='position: absolute; top: 0; right: 0; background: none; border: none; cursor: pointer; font-size: 18px;'>
                        📋
                    </button>
                    <div style='white-space: pre-wrap; word-wrap: break-word; padding-right: 40px;'>{content}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    display_section("🔍 Research", outputs["research"], "research")
    display_section("✍️ Blog Post", outputs["blog"], "blog")
    display_section("📝 Summary", outputs["summary"], "summary")
    display_section("📣 Social Media Content", outputs["social"], "social")
