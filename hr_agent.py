from langchain_ollama import (ChatOllama,OllamaEmbeddings)
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
from langchain_core.chat_history import InMemoryChatMessageHistory
from datetime import datetime
from pydantic import BaseModel              #validate the data got from db
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_classic.agents import (create_tool_calling_agent,AgentExecutor)


llm = ChatOllama(model="qwen2.5:3b")
embedding = OllamaEmbeddings(model = "nomic-embed-text")

vector_db = FAISS.load_local("hr_vector_db" , embedding , allow_dangerous_deserialization = True)       #only give True if u trust the database u r getting from

retriver = vector_db.as_retriever(search_kwargs = {"k":3})

prompt = ChatPromptTemplate.from_template("""You are HR assistant. Answer the candidate question.
                                          Context:{context}
                                          Question:{question}""")

chat_history = InMemoryChatMessageHistory()

@tool
def experience_calculator(start_year:int)->str:
    """Calculate the cadidate experience."""
    return str(datetime.now().year - start_year)

@tool
def eligibility_checker(skills:str)->str:
    """Check candidate skills eligibility."""
    required = {"python" , "sql" , "git"}
    candidate = {skills.strip().lower()
                 for s in skills.split(",")
                    }
    missing = required - candidate
    if len(missing) == 0:
        return "Eligible"
    else:
        return ("Not eligible.Missing:" + ",".join(missing))
    
@tool
def company_policy_search(question:str)->str:
    """Search company policy documents."""
    docs = retriver.invoke(question)
    context = "\n".join(doc.page_content for doc in docs)
    prompt = f"""Answer only from the context given
            Context:{context}
            Question:{question}"""
    result = llm.invoke(prompt)
    return result.content

@tool
def interview_question(skill:str)->str:
    """Generate interview questions."""
    prompt = f"""Generate 5 interview questions
            for:
                {skill}"""
    
#tool
tools = [experience_calculator , eligibility_checker , company_policy_search , interview_question]

#prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an HR Recruitment Assistant.
            Use tools whenever required.

            If the user asks:
            - leave policy notice period
            - Borking hours
            - job description
            - company policy
            Always use company_policy_search"""
        ),
        (
            "human","{input}"
        ),
        (
            "placeholder","{agent_scratchpad}"
        )
    ]
)

agent = create_tool_calling_agent(llm = llm , tools = tools , prompt = prompt)

Agen_Executor = AgentExecutor(agent = agent , tools = tools , verbode = True)

#defining output structure
class candidate(BaseModel):
    name:str
    experience:int
    skills:List[str]

structured_llm = llm.with_structured_output(candidate)


print("=" * 60)
print("RECRUITMENT ASSISTANT")
print("=" * 60)


while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break
    if user_input.startswith("resume:"):
        resume = user_input.replace(
            "resume:",""
        )
        candidate = structured_llm.invoke(
            f"""Extract:
                Name
                Experience
                Skills
                Resume:{resume}"""
        )
        print("\nCandidate Details")
        print(candidate)
    response = Agen_Executor.invoke({"input":user_input})


print("\nAssistant:" , response["output"])