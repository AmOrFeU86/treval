"""
Demo: ReAct agent traced with treval + OpenRouter.
Uses deepseek/deepseek-v4-flash and calls real tools.

Usage:
    source .venv/bin/activate
    OPENROUTER_API_KEY=sk-... python py/demo_react.py "what's the weather in Madrid?"
    OPENROUTER_API_KEY=sk-... python py/demo_react.py "3 * 7 + 12"
"""
import json
import os
import re
import sys

import treval
from treval import agent, operation, tool

# ─── Auto-instrumentation: captures LLM calls without decorating ───
treval.instrument()


# ─── Tools ───

@tool(name="obtener_clima")
def get_weather(ciudad: str) -> str:
    """Gets the current weather for a city."""
    data = {
        "Madrid": "28°C, sunny, light wind",
        "Barcelona": "26°C, partly cloudy",
        "London": "15°C, light rain",
        "Tokyo": "22°C, humid, 70% humidity",
        "Paris": "18°C, cloudy",
        "New York": "24°C, partly cloudy",
    }
    result = data.get(ciudad.capitalize(),
                      f"No weather data for {ciudad}")
    return result


@tool(name="calcular")
def calculate(expresion: str) -> float:
    """Evaluates a mathematical expression. Uses basic operators +, -, *, /."""
    # Only allow safe math
    allowed = re.sub(r'[0-9+\-*/.() ]', '', expresion)
    if allowed:
        raise ValueError(f"Expression contains disallowed characters: {allowed}")
    result = eval(expresion)
    return result


@tool(name="buscar_info")
def search_info(consulta: str) -> str:
    """Searches factual information about a topic."""
    knowledge = {
        "capital of spain": "Madrid is the capital of Spain",
        "spain population": "Spain has approximately 47 million inhabitants",
        "official language": "Spanish (Castilian) is the official language of Spain",
        "spain currency": "The official currency of Spain is the Euro (EUR)",
    }
    query_lower = consulta.lower()
    for key, value in knowledge.items():
        if key in query_lower or query_lower in key:
            return value
    return f"No information found about: {consulta}"


# ─── ReAct Agent ───

TOOLS_DESCRIPTION = """
You have access to these tools:

1. obtener_clima(city) - Gets the current weather for a city
2. calcular(expression) - Evaluates a mathematical expression (+, -, *, /)
3. buscar_info(query) - Searches factual information about a topic

Response format:
- If you need to use a tool, respond EXACTLY like this:
  FUNCION: function_name
  ARGS: {"arg1": "value1", "arg2": "value2"}

- If you already have enough information, respond normally.
"""


@agent(name="ReActBot")
class ReActAgent:
    def __init__(self, api_key: str, model: str = "deepseek/deepseek-v4-flash"):
        from openai import OpenAI
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={
                "HTTP-Referer": "https://treval.dev",
                "X-Title": "treval-demo-react",
            },
        )
        self.model = model

    @operation(name="reason")
    def _call_llm(self, messages: list) -> str:
        """Calls the LLM and returns the response text."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,
            max_tokens=500,
        )
        return response.choices[0].message.content or ""

    def run(self, question: str) -> str:
        """Runs the ReAct loop: think → act → observe → respond."""
        messages = [
            {"role": "system", "content": f"You are a helpful assistant that answers questions using tools.{TOOLS_DESCRIPTION}"},
            {"role": "user", "content": question},
        ]

        max_steps = 5
        for step in range(max_steps):
            response_text = self._call_llm(messages)

            # Detect if the LLM wants to call a tool
            func_match = re.search(r"FUNCION:\s*(\w+)", response_text)
            args_match = re.search(r"ARGS:\s*(\{.+?\})", response_text, re.DOTALL)

            if func_match and args_match:
                func_name = func_match.group(1)
                try:
                    raw_json = args_match.group(1).replace("'", '"')
                    func_args = json.loads(raw_json)
                except json.JSONDecodeError:
                    func_args = {}

                # Execute the tool
                result = self._execute_tool(func_name, func_args)

                # Add to history
                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "user",
                    "content": f"Result of {func_name}: {result}\n\nContinue with the final answer.",
                })
            else:
                # No tool call — this is the final answer
                return response_text

        return "Sorry, could not resolve the question within the maximum number of steps."

    @operation(name="execute_tool")
    def _execute_tool(self, name: str, args: dict) -> str:
        """Executes the requested tool and returns the result."""
        tool_map = {
            "obtener_clima": get_weather,
            "calcular": calculate,
            "buscar_info": search_info,
        }
        fn = tool_map.get(name)
        if not fn:
            return f"Error: tool '{name}' not found"

        try:
            result = fn(**args)
            return str(result)
        except Exception as e:
            return f"Error executing {name}: {e}"


# ─── Main ───

def main():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY is not set")
        print("   Use: OPENROUTER_API_KEY=sk-... python py/demo_react.py 'question'")
        sys.exit(1)

    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "what's the weather in Madrid?"

    print(f"\n{'='*50}")
    print(f"🤖  Treval ReAct Demo")
    print(f"{'='*50}")
    print(f"\n📝 Pregunta: {question}")
    print("─" * 50)

    agent = ReActAgent(api_key=api_key)
    response = agent.run(question)

    print(f"\n💬 Respuesta:\n{response}")
    print(f"\n{'='*50}")
    print(f"📊 Para ver los spans:  treval spans")
    print(f"📊 Detalle de un span: treval span <id>")
    print(f"📊 Para evaluar spans:  treval eval")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()