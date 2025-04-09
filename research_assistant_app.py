import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool  # your custom tools

load_dotenv()

# Pydantic model for structured response
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

# LLM of choice
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Output parser
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a research assistant that will help generate a research paper.
        Answer the user query and use necessary tools. 
        Wrap the output in this format and provide no other text\n{format_instructions}"""),
        ("ai", "{chat_history}"),
        ("human", "{query}"),
        ("ai", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# Tools and agent setup
tools = [search_tool, wiki_tool, save_tool]
agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Streamlit UI
st.title("🧠 Research Assistant")

query = st.text_input("What would you like to research?")
run_button = st.button("Run Research")

if run_button and query:
    with st.spinner("Thinking..."):
        raw_response = agent_executor.invoke({"query": query})

        try:
            structured = parser.parse(raw_response.get("output", raw_response.get("text", "")))
            st.subheader(f"📌 Topic: {structured.topic}")
            st.write(structured.summary)
            st.markdown("**📚 Sources:**")
            for src in structured.sources:
                st.markdown(f"- {src}")
            st.markdown("**🔧 Tools Used:**")
            for tool in structured.tools_used:
                st.markdown(f"- {tool}")
        except Exception as e:
            st.error(f"Error parsing response: {e}")
            st.json(raw_response)
