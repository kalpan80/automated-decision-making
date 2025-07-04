import logging

import asyncio

import numpy as np
from dotenv import load_dotenv
from google.adk import Runner
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel, Field
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

OPENAI_CHATGPT = "gpt-4o"
APP_NAME = "tuning_app"
USER_ID = "12345"
SESSION_ID = "12345678"
session_service = InMemorySessionService()

load_dotenv()


class MLAgent:
    def __init__(self):
        self.model = None

    def get_data(self):
        return load_iris()

    def train(self,split:float, random_state: int, n_estimators: int) -> str:
        print('Model training invoked ',split,random_state, n_estimators)
        iris = self.get_data()
        X,y = iris.data,iris.target

        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=split,random_state=random_state)

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

        self.model.fit(X_train_scaled,y_train)

        y_pred = self.model.predict(self.scaler.transform(X_test))

        return classification_report(y_test,y_pred)

    def infer(self,X_test: list,y_test:list) -> str:
        print('Model inference invoked')
        y_pred = self.model.predict(self.scaler.transform(np.asarray(X_test)))
        return classification_report(np.asarray(y_test),y_pred)


class SyntheticData(BaseModel):
    x_test: list = Field('The test data for the model')
    y_test: list = Field('The ground truth label for the model')


class TuningAgent:

    def __init__(self):
        self.ml_agent = MLAgent()

        self.trainer_agent = LlmAgent(
                name="TrainerAgent",
                description="Agent that invokes model training procedure using tools.",
                model=OPENAI_CHATGPT,
                instruction="You are a training agent. You need to use tools to train the agent."
                            "It expects a split value from 0.5 to 0.8, a random_state is any integer value,"
                                            "and number of estimators for classifier, it should be a integer value from 20 to 50. You need to choose the values carefully for hyper paramater tuning.",
                input_schema=None,
                tools=[self.ml_agent.train]
            )

        self.data_generator_agent = LlmAgent(
                name="SyntheticDataGenerator",
                description="Synthetic data generator agent that creates sample data for testing.",
                model=OPENAI_CHATGPT,
                instruction="You are a synthetic data generator agent. You need to use tools to generate synthetic data for all classes in the dataset.",
                tools=[self.ml_agent.get_data],
                output_key='data'
            )

        self.validator_agent = LlmAgent(
                name="ValidatorAgent",
                description="Validator agent that checks model performance with synthetic data",
                model=OPENAI_CHATGPT,
                instruction="You are a validator agent. You need to use tools to invoke model inference. "
                            "It expects the synthetic data for testing. The synthetic data should be available in the state session with data key",
                input_schema=SyntheticData,
                tools=[self.ml_agent.infer],
            )

        self.planning_agent = LlmAgent(
                name="PlannerAgent",
                model=OPENAI_CHATGPT,
                description="Main planner or orchestrator agent.",
                instruction="You are a planning agent. Your job is to break down complex tasks into smaller manageable subtasks using sub agents."
                            "Your sub agents are as follows:"
                            "TrainerAgent: It trains the machine learning model for inference."
                            "SyntheticDataGenerator: It generates synthetic data for model testing. The synthetic data must be different than the dataset"
                            "ValidatorAgent: It uses the synthetic data to invoke the model inference and generate the test results."
                            "You plan and delegate tasks, and do not execute them."
                            "Please all the agents till accuracy of more than 95% is achieved on test results. You will use only maximum 5 iterations for model retraining."
                            "While assigning tasks use the format:"
                            "1. <agent> : <task>"
                            "After all tasks are completed summarize the findings for the entire process and end with 'TERMINATE'.",
                input_schema=None,
                sub_agents=[self.trainer_agent,self.data_generator_agent,self.validator_agent]
            )



async def call_llm():
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )
    tuning_agent = TuningAgent()
    runner = Runner(
        agent=tuning_agent.planning_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )
    content = types.Content(role='user', parts=[types.Part(text='Select an agent to perform the next task on the basis of current conversation context : {history}.')])

    final_response = None
    async for event in runner.run_async(user_id=USER_ID,session_id=SESSION_ID,new_message=content):
        if event.content and event.content.parts:
            print(event.content.parts[0].text)
            if event.is_final_response():
                final_response = event.content.parts[0].text

asyncio.run(call_llm())