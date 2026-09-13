import streamlit as st
import requests


API_URL = "http://backend:8000/generate"


st.set_page_config(
    page_title="Story Generator",
    page_icon="📖",
    layout="centered"
)


# -----------------------------
# CSS
# -----------------------------

st.markdown(
    """
    <style>

    .title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 30px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# Title
# -----------------------------

st.markdown(
    '<div class="title">📖 Story Generator</div>',
    unsafe_allow_html=True
)


# -----------------------------
# Prompt
# -----------------------------

prompt = st.text_area(
    "Enter your story prompt",
    placeholder="Once upon a time there was a young boy who...",
    height=150,
    key="prompt"
)


# -----------------------------
# Character Counter
# -----------------------------

st.caption(
    f"Characters: {len(prompt)} / 50 minimum"
)


# -----------------------------
# Generate
# -----------------------------

if st.button(
    "Generate Story",
    use_container_width=True
):

    if len(prompt.strip()) < 50:

        st.error(
            "Prompt must be at least 50 characters long."
        )

    else:

        st.subheader("Generated Story")

        story_placeholder = st.empty()

        generated_story = ""

        try:

            with requests.post(
                API_URL,
                json={"prompt": prompt},
                stream=True,
                timeout=120
            ) as response:

                if response.status_code != 200:

                    st.error(
                        response.text
                    )

                else:

                    for token in response.iter_content(
                        chunk_size=None,
                        decode_unicode=True
                    ):

                        if token:

                            generated_story += token

                            story_placeholder.markdown(
                                generated_story
                            )

        except requests.exceptions.RequestException as e:

            st.error(
                f"Could not connect to backend: {e}"
            )