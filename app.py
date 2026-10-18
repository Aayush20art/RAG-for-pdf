        col1, col2 = st.columns(2)
        col1.metric("Chunks", st.session_state.doc_chunks)
        col2.metric("Q&A Turns", len(st.session_state.chat_history))
        st.caption(f"Current PDF: {st.session_state.pdf_name or 'Uploaded PDF'}")

    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

    if st.button("♻️ Reset All"):
        st.session_state.vector_store = None
        st.session_state.embeddings = None
        st.session_state.chat_history = []
        st.session_state.pdf_loaded = False
        st.session_state.doc_chunks = 0
        st.session_state.pdf_name = None
        st.rerun()


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.markdown("# 🔴 RAG PDF Assistant")
st.markdown(
    "Upload a PDF and ask questions about its contents. "
    "Powered by **Groq** and **Hugging Face embeddings**."
)
st.markdown("---")

if not st.session_state.pdf_loaded:
    st.markdown(
        '<div class="rag-card">📌 <b>Getting Started:</b><br>'
        '1. Upload a PDF from the sidebar.<br>'
        '2. Click <b>Process PDF</b>.<br>'
        '3. Ask questions about the document.</div>',
        unsafe_allow_html=True,
    )

if st.session_state.chat_history:
    st.markdown("### 💬 Conversation")
    for turn in st.session_state.chat_history:
        st.markdown("**🙋 You:**")
        st.markdown(turn["user"])
        st.markdown("**🤖 Assistant:**")
        st.markdown(turn["assistant"])

        if turn.get("sources"):
            with st.expander("📎 Source chunks used"):
                for i, source in enumerate(turn["sources"], 1):
                    page = source.metadata.get("page", "?")
                    snippet = source.page_content[:500].replace("\n", " ")
                    st.markdown(
                        f'<div class="rag-card-source">'
                        f'<b>Chunk {i}</b> — Page {page}<br>{snippet}</div>',
                        unsafe_allow_html=True,
                    )
    st.markdown("---")

if st.session_state.pdf_loaded:
    st.markdown("### ❓ Ask a Question")
    with st.form(key="qa_form", clear_on_submit=True):
        user_query = st.text_area(
            "Your question",
            placeholder="e.g. Summarize the document or explain a specific topic.",
            height=100,
        )
        submitted = st.form_submit_button("🔍 Ask")

    if submitted:
        if not user_query.strip():
            st.warning("Please enter a question.")
        else:
            try:
                with st.spinner("Retrieving context and generating answer..."):
                    sources = retrieve_context(user_query, k=top_k)
                    context = "\n\n".join(
                        doc.page_content for doc in sources
                    )
                    if not context.strip():
                        st.warning("No relevant text was retrieved from the PDF.")
                    else:
                        answer = ask_groq(
                            query=user_query.strip(),
                            context=context,
                            model=groq_model,
                            history=st.session_state.chat_history,
                        )
                        st.session_state.chat_history.append({
                            "user": user_query.strip(),
                            "assistant": answer,
                            "sources": sources,
                        })
                        st.rerun()
            except RuntimeError as exc:
                st.error(str(exc))
            except Exception as exc:
                print(f"Unexpected application error: {type(exc).__name__}: {exc}")
                st.error(
                    "An unexpected error occurred while generating the answer. "
                    "Open Streamlit Cloud → Manage app → Logs for details."
                )
else:
    st.info("⬅️ Upload and process a PDF from the sidebar to start chatting.")
