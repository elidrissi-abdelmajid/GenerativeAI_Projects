from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import OpenAI
from dotenv import load_dotenv
import os
import streamlit as st

# Load the OpenAI API key from the environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def main():
    load_dotenv()

    if OPENAI_API_KEY is None or OPENAI_API_KEY == "":
        print("OPENAI_API_KEY is not set")
        # st.set_page_config(page_title="open api key not seting !")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")

    st.set_page_config(page_title="Ask your CSV 📈")
    st.header("Ask your CSV 📈")

    csv_file = st.file_uploader("Upload a CSV file", type="csv")
    if csv_file is not None:
        user_question = st.text_input("Ask a question about your CSV: ")
        agent = create_csv_agent(
            OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0),
            csv_file,
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        )

        if user_question is not None and user_question != "":
            # with st.spinner(text="In progress..."):
                # st.write(agent.run(user_question))
            st.write(f'your question was  : **{user_question} **')
            response=agent.run(user_question)
            st.write(response)

if __name__ == "__main__":
    main()
