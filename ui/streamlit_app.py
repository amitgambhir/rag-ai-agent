import streamlit as st
from modules.rag_qa import RAGQA
from dotenv import load_dotenv
import os

load_dotenv()

USERNAME = os.getenv("STREAMLIT_AUTH_USERNAME", "admin")
PASSWORD = os.getenv("STREAMLIT_AUTH_PASSWORD", "password")

def check_password():
    def password_entered():
        if (
            st.session_state.get("username") == USERNAME and
            st.session_state.get("password") == PASSWORD
        ):
            st.session_state["authenticated"] = True
            # Clear password from session state
            del st.session_state["password"]
        else:
            st.error("Invalid username or password")

    if "authenticated" not in st.session_state:
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password", on_change=password_entered)
        return False
    elif not st.session_state["authenticated"]:
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password", on_change=password_entered)
        return False
    else:
        # Show logout button
        if st.button("Logout"):
            for key in ["authenticated", "username"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.experimental_rerun()
        return True

def main():
    st.title("AI Agent MCP Chat")

    if not check_password():
        return

    rag = RAGQA()

    query = st.text_input("Ask me anything related to your documents:")

    if query:
        with st.spinner("Thinking..."):
            answer, sources = rag.query(query)
        st.markdown(f"**Answer:** {answer}")
        st.markdown(f"**Sources:** {', '.join(sources)}")

if __name__ == "__main__":
    main()
