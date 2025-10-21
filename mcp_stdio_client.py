from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai import usage as _usage
import asyncio
import logging
import os

# Ensure the server subprocess runs with unbuffered output so prints/logging flow to parent immediately
env = os.environ.copy()
# PYTHONUNBUFFERED=1 forces the child Python process to use unbuffered binary stdout and stderr
env.setdefault('PYTHONUNBUFFERED', '1')

server = MCPServerStdio(
    'python', args=['server.py'], timeout=10, env=env
)

# Enable debug logging in client so tool calls and responses are visible in the terminal
logging.basicConfig(level=logging.DEBUG)
agent = Agent('openai:gpt-4.1-mini', toolsets=[server])


async def main():
    async with agent:
        # Increase the request_limit so the run can make more model requests before hitting the default
        usage_limits = _usage.UsageLimits(request_limit=200)
        result = await agent.run(
            "Can you download the data for the 'home-data-for-ml-course' competition on Kaggle, drop all categorical columns, impute missing values, create a HistGradientBoostingRegressor model, and submit a prediction? Do this is as few steps as possible.",
            usage_limits=usage_limits,
        )
    print(result.output)

if __name__ == "__main__":
    asyncio.run(main())