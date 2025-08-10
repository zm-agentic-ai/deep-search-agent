import os
import asyncio
from dotenv import load_dotenv, find_dotenv
from dataclasses import dataclass
from typing import Callable
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled, function_tool, RunContextWrapper
from openai.types.responses import ResponseTextDeltaEvent
from tavily import TavilyClient

# Load environment variables
_ = load_dotenv(find_dotenv())

gemini_api_key: str | None = os.environ.get("GEMINI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=tavily_api_key)

set_tracing_disabled(disabled=True)

# LLM Client
external_client: AsyncOpenAI = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Model
llm_model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_client
)

@dataclass
class UserContext:
    username: str
    email: str
    history_file: str

# Tool for Tavily Search
@function_tool
def searchTavily(query: str) -> str:
    print(f"Searching for {query}...")
    response = tavily_client.search(query)
    return response

# Agent Instructions
def special_prompt(special_context: RunContextWrapper[UserContext], agent: Agent[UserContext]) -> str:
    return (f"You are a special deep search agent that can search the web for information. "
            f"User Name is : {special_context.context.username}, Agent: {agent.name}. "
            f"Please do a deep search for the {special_context.context.username} query.")

# Agent
deep_search_agent: Agent = Agent(
    name="Deep Search Agent",
    instructions=special_prompt,
    model=llm_model,
    tools=[searchTavily]
)

# Create or load user history file
def load_or_create_user_file(email: str, username: str) -> str:
    safe_email = email.replace("@", "_at_").replace(".", "_dot_")
    file_path = f"user_history_{safe_email}.txt"

    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"User: {username} | Email: {email}\n")
            f.write("Prompt History:\n")
        print(f"Created new history file: {file_path}")
    else:
        print(f"Loaded existing history file: {file_path}")

    return file_path

# Append prompt to user history
def save_prompt_to_history(file_path: str, prompt: str):
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"- {prompt}\n")

# View history
def view_history(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        print("\n Prompt History:\n")
        print(f.read())

# Main Agent Runner
async def call_agent():
    # Step 1: Get user details
    username = input("Enter your name: ").strip()
    email = input("Enter your email: ").strip()

    # Step 2: Load or create history file
    history_file = load_or_create_user_file(email, username)

    # Step 3: Create user context
    user_context = UserContext(username=username, email=email, history_file=history_file)

    while True:
        print("\nChoose an option:")
        print("1. View Prompt History")
        print("2. Search for New Prompt")
        print("3. Exit")
        choice = input("Enter choice (1/2/3): ").strip()

        if choice == "1":
            view_history(history_file)

        elif choice == "2":
            query = input("Enter your search query: ").strip()
            save_prompt_to_history(history_file, query)

            print("\n🔍 Search Result:\n")

            # Run agent and stream output
            output = Runner.run_streamed(
                starting_agent=deep_search_agent,
                input=query,
                context=user_context
            )

            async for event in output.stream_events():
                if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                    print(event.data.delta, end="", flush=True)
            print()  # New line after streaming

        elif choice == "3":
            print("Goodbye!")
            break

        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    asyncio.run(call_agent())
