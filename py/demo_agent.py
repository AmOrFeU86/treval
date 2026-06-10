"""
Demo: simple agent traced with treval + OpenRouter.

Usage:
    OPENROUTER_API_KEY=sk-... python py/demo_agent.py "what's the weather in Madrid"
"""
import os
import sys
from openai import OpenAI

from treval import agent, operation, tool


# --- Tools ---

@tool(name="buscar_clima")
def get_weather(city: str) -> str:
    """Simulates fetching weather for a city."""
    data = {
        "Madrid": "28°C, sunny",
        "Barcelona": "26°C, cloudy",
        "London": "15°C, rain",
        "Tokyo": "22°C, humid",
        "Paris": "18°C, overcast",
    }
    result = data.get(city, f"Weather not available for {city}")
    return result


@tool(name="calcular")
def calculate(expression: str) -> float:
    """Evaluates a simple mathematical expression."""
    return eval(expression)


# --- Agent ---

@agent(name="WeatherBot")
class WeatherBot:
    def __init__(self, api_key: str):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={
                "HTTP-Referer": "https://treval.dev",
                "X-Title": "treval-demo",
            },
        )

    @operation(name="pensar")
    def think(self, messages: list) -> str:
        """Calls the LLM to decide what to do."""
        response = self.client.chat.completions.create(
            model="deepseek/deepseek-v4-flash",
            messages=messages,
            temperature=0.1,
            max_tokens=300,
        )
        return response.choices[0].message.content or ""


def run_agent(question: str):
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY is not set")
        print("Use: OPENROUTER_API_KEY=sk-... python py/demo_agent.py 'question'")
        sys.exit(1)

    bot = WeatherBot(api_key)

    system_prompt = """You are a helpful assistant. You have access to these tools:
- buscar_clima(city): gets the weather for a city
- calcular(expression): evaluates a mathematical expression

Respond naturally and concisely."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    # LLM call
    print(f"\n🤖 Question: {question}")
    print("🤔 Thinking...")
    response = bot.think(messages)

    print(f"\n📝 Answer: {response}")
    print("\n✅ Done. Spans are in ~/.treval/spans.db")
    print("   To view them: treval spans")


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "what's the weather in Madrid?"
    run_agent(question)