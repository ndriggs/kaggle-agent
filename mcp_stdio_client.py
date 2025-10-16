from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
import asyncio

server = MCPServerStdio(  
    'python', args=['server.py'], timeout=10
)
agent = Agent('openai:gpt-4o', toolsets=[server])


async def main():
    async with agent:
        result = await agent.run("Can you download the data for the 'home-data-for-ml-course' competition on Kaggle, create a regression model, and submit a prediction?")
    print(result.output)

if __name__ == "__main__":
    asyncio.run(main())