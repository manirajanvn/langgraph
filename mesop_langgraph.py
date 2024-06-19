
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_community.llms import Ollama
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph
import mesop as me
import mesop.labs as mel

ollm =Ollama(model="llama3")

#### Langchains
prompt = PromptTemplate(
    template="""You are an math and science expert at routing a user question to a math or science. \n
    understand the content and find the question is relavant to maths or science. \n
    Give a binary choice 'math' or 'science' based on the question. \n
    Return the a JSON with a single key 'subject' and no premable or explanation. \n
    Question to route: {question}""",
    input_variables=["question"],
)
sub_router = prompt | ollm | JsonOutputParser()

math_prompt = PromptTemplate(
    template="""You are an excellent math professor that likes to solve math questions in a way that everyone can understand your solution
    Provide the solution to the students that are asking below mathematical questions and give them the answer. \n
    Return process to solve the given maths problem. \n
    Question: {question}""",
    input_variables=["question"],
)
math_chain = math_prompt | ollm | StrOutputParser()

science_prompt = PromptTemplate(
    template="""You are an excellent science professor that likes to solve science questions in a way that everyone can understand your solution
    Provide the solution to the students that are asking below science questions and give them the answer. \n
    Return process to solve the given science problem. \n
    Question: {question}""",
    input_variables=["question"],
)
science_chain = science_prompt | ollm | StrOutputParser()

story_prompt = PromptTemplate(
    template=""" You are an creative fictional story writer and create interesting story for given mathemetical or science answer in a way that everyone can like it
    Create the small story for the kids with the below mathematical or science answer\n
    Answer: {answer}""",
    input_variables=["answer"],
)
story_chain = story_prompt | ollm | StrOutputParser()

###State
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        answer: LLM generation
        story: story
    """

    question: str
    answer: str
    story: str

def math_agent(state):
    question = state["question"]
    print("Math Agent: "+question)
    answer = math_chain.invoke({"question": question})
    return {"answer": answer}


def science_agent(state):
    question = state["question"]
    print("Science Agent: "+question)
    answer = science_chain.invoke({"question": question})
    return {"answer": answer}

def story_agent(state):
    answer = state["answer"]
    print("Executing Story agent")
    story = story_chain.invoke({"answer": answer})
    return {"story": story}

def route_question(state):
    question = state["question"]
    source = sub_router.invoke({"question": question})
    if source["subject"] == "math":
        return "math"
    elif source["subject"] == "science":
        return "science"


workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("math", math_agent) 
workflow.add_node("science", science_agent)  
workflow.add_node("write_story", story_agent)  

# Build graph
workflow.set_conditional_entry_point(
    route_question,
    {
        "math": "math",
        "science": "science",
    },
)
workflow.add_edge("math", "write_story")
workflow.add_edge("science", "write_story")
workflow.add_edge("write_story", END)
# Compile
app = workflow.compile()


@me.page(
  security_policy=me.SecurityPolicy(
    allowed_iframe_parents=["https://google.github.io"]
  ),
  path="/chat",
  title="Story Chat",
)
def page():
  mel.chat(transform, title="Story Chat", bot_user="Langgraph Bot")


def transform(prompt: str, history: list[mel.ChatMessage]) -> str:
  inputs = {
    "question":  prompt
  }
  final_states = app.invoke(inputs)
  return final_states["story"]