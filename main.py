import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory

# 从组件模块导入
from components.task_analysis import create_task_analysis_chain
from components.search import create_search_chain
from components.analysis import create_analysis_chain
from components.decision import create_decision_chain
from components.modification import create_modification_chain
from components.summarize import create_summarize_chain

# 加载环境变量
load_dotenv()

def create_service_lad_agent():
    # 初始化LLM
    llm = ChatOpenAI(temperature=0.1)
    
    # 创建共享记忆
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # 初始化各个组件链
    task_analysis_chain = create_task_analysis_chain(llm)
    search_chain = create_search_chain(llm)
    analysis_chain = create_analysis_chain(llm)
    decision_chain = create_decision_chain(llm)
    modification_chain = create_modification_chain(llm)
    summarize_chain = create_summarize_chain(llm)
    
    # 创建主工作流
    workflow = SequentialChain(
        chains=[
            task_analysis_chain,
            search_chain,
            analysis_chain,
            decision_chain,
            # 这里需要添加条件逻辑，后面会实现
        ],
        input_variables=["input", "current_design"],
        output_variables=["tasks", "queries", "search_results", "analysis_results", 
                        "modification_needed", "modification_results", "summary", "output"],
        verbose=True
    )
    
    return workflow

if __name__ == "__main__":
    agent = create_service_lad_agent()
    result = agent.invoke({
        "input": "I need to improve the login page UX",
        "current_design": "Current login page has username/password fields and a login button"
    })
    print(result)