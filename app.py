# ANALÍTICA BOUTIQUE, SC (https://www.visoresanalitica.com.mx/)
# DEMO
# 
# PROBLEM: 

# Multi-agent Collaboration for Financial Analysis
# Demostrate ways for making agents collaborate with each other.
# 

# streamlit run app.py

# Dependencies:
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import os
import sys
import io
import streamlit as st
import pandas as pd
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
#from scrape_website_tool import ScrapeWebsiteTool
#from serper_dev_tool import SerperDevTool

# 
# Configuración de la App
TITLE = "Colaboración Multiagente para el Análisis Financiero"
st.set_page_config(page_title = TITLE, page_icon = "📊", )
st.title(TITLE)

# **Agregar Imagen en la Barra Lateral**
st.sidebar.image("images/Logo_AB.png", use_container_width=True)  # Ruta a la imagen
st.sidebar.markdown("Contact: vicente@analiticaboutique.com.mx benjamin@analiticaboutique.com.mx")
# Barra lateral - Selección del modelo de OpenAI
st.sidebar.title("Configuración del Modelo")

MODEL_LIST = ['gpt-4o-mini', 'gpt-4o']

# * OpenAI Model Selection
model_option = st.sidebar.selectbox(
    "Choose OpenAI model",
    MODEL_LIST,
    index=0
)

#
os.environ["OPENAI_MODEL_NAME"] = model_option
# Obtener las API keys desde .env
#openai_api_key = os.getenv("OPENAI_API_KEY")
#serper_api_key = os.getenv("SERPER_API_KEY")

# Obtener las API keys desde SECRETS
openai_api_key = st.secrets["OPENAI_API_KEY"]
serper_api_key = st.secrets["SERPER_API_KEY"]

# Descripción de la app
st.title("📊 AI Agents para Análisis de Trading")
st.markdown("""
**Este DEMO crea un sistema de agentes de inteligencia artificial (AI Agents) para analizar datos del mercado financiero y sugerir estrategias de trading para un activo.**

La tripulación de agentes incluye a:

* Data Analyst

* Trading Strategy Developer

* Trade Advisor

* Risk Advisor

Todos, con herramientas de acceso a información en tiempo real que sea disponible en Internet.

Los supuestos considerados en la estrategia son:

* Un capital inicial de 100,000 USD;

* Tolerancia al riesgo media, y

* Una preferencia de estrategia de trading diaria.

Este DEMO muestra el proceso de razonamiento seguido por los agentes para llegar al resultado final.

Por favor, selecciona el **Ticker** respecto del cual quieras el análisis financiero.

Por ejemplo:

* VOD - Índice Nasdaq

* AAPL - Apple

* GOOG - Alphabet (Google)

* INTC - Intel

* BTC-USD - Bitcoin USD    
""")

# Entrada del usuario: Selección del Ticker
ticker = st.text_input("Introduce el Ticker del activo financiero (Ej: AAPL):", value="AAPL")

# crewAI Tools
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

# Creating Agents
# Agent: Data Analyst
data_analyst_agent = Agent(
    role="Data Analyst",
    goal="Monitor and analyze market data in real-time "
         "to identify trends and predict market movements.",
    backstory="Specializing in financial markets, this agent "
              "uses statistical modeling and machine learning "
              "to provide crucial insights. With a knack for data, "
              "the Data Analyst Agent is the cornerstone for "
              "informing trading decisions.",
    verbose=True,
    allow_delegation=True,
    tools = [scrape_tool, search_tool]
)
# Agent: Trading Strategy Developer 
trading_strategy_agent = Agent(
    role="Trading Strategy Developer",
    goal="Develop and test various trading strategies based "
         "on insights from the Data Analyst Agent.",
    backstory="Equipped with a deep understanding of financial "
              "markets and quantitative analysis, this agent "
              "devises and refines trading strategies. It evaluates "
              "the performance of different approaches to determine "
              "the most profitable and risk-averse options.",
    verbose=True,
    allow_delegation=True,
    tools = [scrape_tool, search_tool]
)
# Agent: Trade Advisor
execution_agent = Agent(
    role="Trade Advisor",
    goal="Suggest optimal trade execution strategies "
         "based on approved trading strategies.",
    backstory="This agent specializes in analyzing the timing, price, "
              "and logistical details of potential trades. By evaluating "
              "these factors, it provides well-founded suggestions for "
              "when and how trades should be executed to maximize "
              "efficiency and adherence to strategy.",
    verbose=True,
    allow_delegation=True,
    tools = [scrape_tool, search_tool]
)
# Agent: Risk Advisor
risk_management_agent = Agent(
    role="Risk Advisor",
    goal="Evaluate and provide insights on the risks "
         "associated with potential trading activities.",
    backstory="Armed with a deep understanding of risk assessment models "
              "and market dynamics, this agent scrutinizes the potential "
              "risks of proposed trades. It offers a detailed analysis of "
              "risk exposure and suggests safeguards to ensure that "
              "trading activities align with the firm’s risk tolerance.",
    verbose=True,
    allow_delegation=True,
    tools = [scrape_tool, search_tool]
)

# Creating Tasks
# Task for Data Analyst Agent: Analyze Market Data
data_analysis_task = Task(
    description=(
        "Continuously monitor and analyze market data for "
        "the selected stock ({stock_selection}). "
        "Use statistical modeling and machine learning to "
        "identify trends and predict market movements."
    ),
    expected_output=(
        "Insights and alerts about significant market "
        "opportunities or threats for {stock_selection}."
    ),
    agent=data_analyst_agent,
)

# Task for Trading Strategy Agent: Develop Trading Strategies
strategy_development_task = Task(
    description=(
        "Develop and refine trading strategies based on "
        "the insights from the Data Analyst and "
        "user-defined risk tolerance ({risk_tolerance}). "
        "Consider trading preferences ({trading_strategy_preference})."
    ),
    expected_output=(
        "A set of potential trading strategies for {stock_selection} "
        "that align with the user's risk tolerance."
    ),
    agent=trading_strategy_agent,
)

# Task for Trade Advisor Agent: Plan Trade Execution
execution_planning_task = Task(
    description=(
        "Analyze approved trading strategies to determine the "
        "best execution methods for {stock_selection}, "
        "considering current market conditions and optimal pricing."
    ),
    expected_output=(
        "Detailed execution plans suggesting how and when to "
        "execute trades for {stock_selection}."
    ),
    agent=execution_agent,
)

# Task for Risk Advisor Agent: Assess Trading Risks
risk_assessment_task = Task(
    description=(
        "Evaluate the risks associated with the proposed trading "
        "strategies and execution plans for {stock_selection}. "
        "Provide a detailed analysis of potential risks "
        "and suggest mitigation strategies."
    ),
    expected_output=(
        "A comprehensive risk analysis report detailing potential "
        "risks and mitigation recommendations for {stock_selection}."
        "Provide your final answer in Spanish."
    ),
    agent=risk_management_agent,
)

# Creating the Crew
# Note: The Process class helps to delegate the workflow to the Agents (kind of like a Manager at work)
#       In this example, it will run this hierarchically.
#       manager_llm lets you choose the "manager" LLM you want to use.

# Define the crew with agents and tasks
financial_trading_crew = Crew(
    agents=[data_analyst_agent,
            trading_strategy_agent,
            execution_agent,
            risk_management_agent],

    tasks=[data_analysis_task,
           strategy_development_task,
           execution_planning_task,
           risk_assessment_task],

    manager_llm=ChatOpenAI(model=model_option,
                           temperature=0.7),
    process=Process.hierarchical,
    verbose=True
)

# Botón para iniciar la ejecución
if st.button("🚀 Iniciar Análisis"):
    st.write("⏳ Ejecutando análisis con AI Agents... Esto puede tardar unos segundos.")

    # Definir los inputs para la Crew
    financial_trading_inputs = {
        'stock_selection': ticker,
        'initial_capital': '100000',
        'risk_tolerance': 'Medium',
        'trading_strategy_preference': 'Day Trading',
        'news_impact_consideration': True
    }

    # Capturar la salida de la ejecución
    output_buffer = io.StringIO()
    sys.stdout = output_buffer  # Redirigir stdout para capturar en tiempo real

    # Placeholder en Streamlit para actualizar el proceso en vivo
    process_placeholder = st.empty()

    # Ejecutar la Crew y actualizar el proceso en vivo
    result = financial_trading_crew.kickoff(inputs=financial_trading_inputs)

    # Restaurar la salida estándar
    sys.stdout = sys.__stdout__

    # Mostrar el resultado final
    st.subheader("📌 Resultado Final")
    st.markdown(result)

    # Mostrar el proceso capturado en tiempo real
    process_placeholder.text_area("🔎 Detalle del Proceso de Razonamiento", value=output_buffer.getvalue(), height=300)
    
