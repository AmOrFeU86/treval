"""ReAct agent traced with treval — works right out of the box!"""
import os, json, re, sys
from openai import OpenAI

import treval
from treval import agent, operation, tool

treval.instrument()

TOOLS = {}


@tool(name="get_weather")
def get_weather(city: str) -> str:
    """Gets the current weather for a city."""
    data = {
        "Madrid": "28°C, sunny",
        "Barcelona": "26°C, cloudy",
        "London": "15°C, rainy",
        "Tokyo": "22°C, humid",
        "New York": "24°C, partly cloudy",
    }
    return data.get(city.capitalize(), f"No weather data for {city}")


TOOLS["get_weather"] = get_weather


@tool(name="calculate")
def calculate(expression: str) -> float:
    """Evaluates a mathematical expression."""
    allowed = re.sub(r"[0-9+\-*/.() ]", "", expression)
    if allowed:
        raise ValueError(f"Disallowed characters: {allowed}")
    return eval(expression)


TOOLS["calculate"] = calculate


@agent(name="ReActBot")
class ReActAgent:
    def __init__(self, api_key: str):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={"HTTP-Referer": "https://treval.dev", "X-Title": "treval-agent"},
        )
        self.model = "deepseek/deepseek-v4-flash"

    @operation(name="reason")
    def _call_llm(self, messages: list) -> str:
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=0.1, max_tokens=500,
        )
        return resp.choices[0].message.content or ""

    def run(self, question: str) -> str:
        tools_desc = "\n".join(
            f"- {name}({', '.join(fn.__code__.co_varnames)}): {fn.__doc__}"
            for name, fn in TOOLS.items()
        )
        messages = [
            {"role": "system", "content": f"You are a ReAct assistant. Tools:\n{tools_desc}\nFormat:\nFUNCTION: name\nARGS: {{}}\n\nIf you have the answer, respond directly."},
            {"role": "user", "content": question},
        ]
        for _ in range(5):
            response = self._call_llm(messages)
            func_match = re.search(r"FUNCTION:\s*(\w+)", response)
            args_match = re.search(r"ARGS:\s*(\{.+?\})", response, re.DOTALL)
            if func_match and args_match:
                func_name = func_match.group(1)
                try:
                    func_args = json.loads(args_match.group(1).replace("'", '"'))
                    fn = TOOLS.get(func_name)
                    result = fn(**func_args) if fn else f"Tool not found: {func_name}"
                except Exception as e:
                    result = f"Error: {e}"
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"Result: {result}\n\nContinue."})
            else:
                return response
        return "Could not resolve the question."


def main():
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENROUTER_API_KEY or OPENAI_API_KEY")
        sys.exit(1)

    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What's the weather in Madrid?"
    agent = ReActAgent(api_key)
    print(f"\nQuestion: {question}")
    response = agent.run(question)
    print(f"\n{response}")
    print("\nTo view traces: treval spans")
    print("To evaluate:    treval eval")


if __name__ == "__main__":
    main()