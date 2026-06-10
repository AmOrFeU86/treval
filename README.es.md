# treval ⚡

<p align="center">
  <a href="https://pypi.org/project/treval/"><img src="https://img.shields.io/pypi/v/treval?color=blue" alt="PyPI version"></a>
  <a href="https://pypi.org/project/treval/"><img src="https://img.shields.io/pypi/pyversions/treval" alt="Python versions"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/AmOrFeU86/treval" alt="License: MIT"></a>
</p>

<p align="center">
  <picture>
    <img alt="Logo de treval" src="https://raw.githubusercontent.com/AmOrFeU86/treval/main/treval_logo_2.png" width="200" height="200">
  </picture>
</p>

[![en](https://img.shields.io/badge/lang-en-red.svg)](README.md)
[![es](https://img.shields.io/badge/lang-es-blue.svg)](README.es.md)

> Traza, evalúa y mejora agentes de IA desde la terminal.

Treval es un framework de observabilidad y evaluación para agentes de IA. Con una sola línea (`import treval; treval.instrument()`) obtienes trazado completo de cada llamada LLM, herramienta y operación. Además: evaluación LLM-as-judge, comparación multi-modelo con estadísticas, costes de API, reproducción de spans, tests nativos para agentes, dashboard web, exportación OpenTelemetry e informes HTML autónomos.

---

## Instalación

```bash
pip install treval
```

**Dependencias:** `openai`, `rich` (el resto son stdlib de Python 3.11+).

Necesitas una API key de [OpenRouter](https://openrouter.ai/keys/) (o de OpenAI si usas OpenAI directamente).

```bash
# En tu ~/.bashrc o antes de ejecutar treval
export OPENROUTER_API_KEY=sk-or-v1-...
```

```bash
# Verifica la instalación
treval --help           # 15 comandos disponibles
treval prices           # Precios actualizados de OpenRouter
```

---

## Trazado Básico

### Auto-instrumentación (una línea)

```python
import treval

treval.instrument()   # Parchea OpenAI síncrono/asíncrono → spans LLM automáticos

# A partir de aquí, TODAS las llamadas a OpenAI se trazan automáticamente
```

### Decorador @agent

```python
from treval import agent, operation, tool

@agent(name="WeatherBot")
class WeatherAgent:
    def __init__(self, api_key: str):
        from openai import OpenAI
        # OpenRouter como proveedor por defecto
        self.client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    @operation
    def get_forecast(self, city: str) -> str:
        """Cada llamada @operation se registra como un span hijo del agente."""
        return self._call_llm(f"weather in {city}")

    @operation(name="call_llm")
    def _call_llm(self, prompt: str) -> str:
        """Las llamadas LLM via OpenAI se trazan automáticamente si llamaste a instrument()."""
        resp = self.client.chat.completions.create(
            model="deepseek/deepseek-v4-flash",
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content
```

### Decorador @tool

```python
@tool(name="get_weather")
def get_weather(city: str) -> str:
    """Cada herramienta se registra como un span de tipo TOOL."""
    return f"28°C, sunny in {city}"
```

### Ver Spans

```bash
treval spans                # Lista los 20 spans más recientes
treval spans -t LLM         # Solo spans LLM
treval spans -l 50          # 50 spans
treval span 42              # Detalle completo del span (input, output, hijos)
treval metrics              # Métricas agregadas por tipo
treval count                # Total de spans almacenados
treval clear                # Borra todos los spans
```

Los spans tienen 4 tipos, representados como insignias de color en el dashboard:

| Tipo | Color | Significado |
|------|-------|-------------|
| **AGENT** | 🔵 Azul | Instancia completa de un agente |
| **OPERATION** | 🟢 Verde | Operación dentro del agente |
| **TOOL** | 🟡 Amarillo | Herramienta o función ejecutada |
| **LLM** | 🟣 Púrpura | Llamada a un modelo de lenguaje |

Los spans se organizan en una jerarquía padre → hijo automáticamente mediante `parent_id`.

---

## Evaluación LLM-as-Judge

```bash
# Evalúa spans recientes con DeepSeek como juez
treval eval                             # Por defecto: correctness
treval eval -c conciseness              # Concisión
treval eval -c helpfulness              # Utilidad
treval eval -t LLM -c correctness       # Solo spans LLM
treval evals                            # Historial de evaluaciones
```

También desde Python:

```python
from treval import LLMEvaluator, EvalStore

evaluator = LLMEvaluator(
    model="deepseek/deepseek-v4-flash",
    criteria="The response must be correct and helpful",
)
results = evaluator.evaluate(spans)

store = EvalStore()
store.save(results[0])
stats = store.get_stats()  # media, min, max
```

El juez usa un parser JSON tolerante que maneja JSON mal formado (strings sin cerrar, markdown, texto extra). Si falla, reintenta automáticamente hasta 2 veces.

---

## Comparación de Modelos (`treval compare`)

Compara **N modelos** con el **mismo prompt**, cada uno **M veces**, con estadísticas (media σ) y costes reales de la API de OpenRouter.

```bash
# 2 modelos, 3 ejecuciones cada uno
treval compare \
  -p "Explain the difference between CNN and Transformer" \
  -m deepseek/deepseek-v4-flash,deepseek/deepseek-v4-pro \
  -r 3

# 4 modelos, 5 ejecuciones, exportar a HTML
treval compare \
  -p "what is fine-tuning?" \
  -m deepseek/deepseek-v4-flash,deepseek/deepseek-v4-pro,anthropic/claude-sonnet-4,xiaomi/mimo-v2.5-pro \
  -r 5 \
  -o comparison.html

# Con criterio personalizado
treval compare -p "summarize this" -m m1,m2 -c conciseness
```

**Salida en terminal:** tabla con #, modelo, puntuación media, σ, duración, coste/ejec., tokens, ejecuciones. Ganador marcado con 🏆.

**HTML exportado** incluye:
- Banner del ganador con puntuación
- Tabla resumen ordenable
- Detalle por modelo con cada ejecución individual
- Output expandible por ejecución
- **Árbol de trazas** (modo agente): jerarquía completa de spans con tipos coloreados

### Modo Agente

Compara ejecuciones completas de un script de agente instrumentado con treval:

```bash
treval compare --agent "python my_agent.py 'question'" -r 5 -o agents.html
```

Cada ejecución:
1. Ejecuta el script como subproceso
2. Captura stdout (como output)
3. Lee los nuevos spans que el agente guardó en la BD
4. Evalúa el output con LLM-as-judge
5. Renderiza el **árbol de trazas jerárquico** en el HTML

---

## Replay (`treval replay`)

Re-ejecuta un span guardado cambiando modelo, temperatura o input:

```bash
treval replay 42                          # Re-ejecuta con los mismos parámetros
treval replay 42 --model anthropic/claude-sonnet-4  # Cambia el modelo
treval replay 42 --input "new question"              # Cambia el input
treval replay 42 --temperature 0.5                   # Cambia la temperatura
```

Muestra una tabla comparativa: output original vs nuevo, duración y uso de tokens.

---

## Tests de Agentes

Define tests para agentes usando LLM-as-judge:

```python
# tests/test_my_agent.py
from treval.testing import case, TestSuite

suite = TestSuite(name="WeatherTests")

@case(suite,
      input="What's the weather like in Madrid?",
      criteria="The response must mention Madrid's weather")
def test_madrid(response: str) -> None:
    assert "Madrid" in response
    assert "28" in response or "sunny" in response
```

```bash
treval test run tests/test_my_agent.py
```

Cada test ejecuta el agente, evalúa el output con LLM-as-judge y muestra ✅/❌ con puntuación y razón.

---

## Dashboard

```bash
treval dashboard                     # Servidor web en http://127.0.0.1:8080
treval dashboard --port 3000         # Puerto personalizado
treval dashboard --no-open           # No abre el navegador
treval dashboard --export report.html  # HTML autónomo (funciona desde file://)
```

El dashboard exportado es 100% autónomo (sin servidor), responsive, con:
- Estadísticas (total, agentes, operaciones, herramientas, LLMs, errores)
- Tabla ordenable por cualquier columna
- Panel de detalle con input/output y jerarquía de hijos
- Barras de duración con código de color
- Leyenda de tipos de span
- Diseño oscuro adaptado a móvil

---

## Gateway Proxy

Intercepta tráfico LLM para trazarlo sin modificar código:

```bash
treval gateway                       # Proxy en :9090 → OpenRouter
treval gateway --port 9090 --upstream openai   # → OpenAI
```

Útil para agentes que no puedes modificar: apunta sus llamadas al gateway y treval registra todo.

---

## Exportación OpenTelemetry

```bash
treval export --console              # Exporta spans a consola (formato OTEL)
treval export --endpoint http://localhost:4317  # Envía a collector OTEL
```

---

## Comparación A/B (legacy)

```bash
treval ab "my question" --model-a flash --model-b pro
```

Comparación simple de 2 modelos sobre el mismo input. Se recomienda usar `treval compare` para 2+ modelos con estadísticas.

---

## Precios en Tiempo Real (`treval prices`)

Obtiene los precios actualizados de la API de OpenRouter automáticamente, sin hardcodear:

```bash
treval prices                          # Todos los modelos disponibles
treval prices --search flash           # Filtra por nombre
treval prices --search deepseek        # Solo modelos DeepSeek
treval prices --search xiaomi          # Solo Xiaomi MiMo
```

Los precios se cachean durante 1 hora en memoria. Si la API no responde, se usa un fallback local con ~20 modelos comunes. Los costes en `treval compare` usan estos precios automáticamente.

---

## API Pública (Python)

```python
import treval

# Decoradores
treval.instrument()               # Auto-instrumentación OpenAI
treval.agent                      # @treval.agent — marca una clase como agente
treval.operation                  # @treval.operation — marca un método como operación
treval.tool                       # @treval.tool — marca una función como herramienta
treval.wrap(client)               # Envuelve un cliente OpenAI existente
treval.wrap_anthropic(client)     # Envuelve un cliente Anthropic existente

# Evaluación
treval.LLMEvaluator               # Evaluador LLM-as-judge
treval.EvalStore                  # Almacén de evaluaciones SQLite

# Callbacks
treval.trace                      # Callback de trazado
treval.on_tool_start / on_tool_end
treval.on_llm_start / on_llm_end

# Comparación (desde Python)
from treval.compare import compare_models, compare_agents, build_report_html
results = compare_models(prompt="...", models=["m1", "m2"], runs=3)
html = build_report_html(results, prompt="...", criteria="correctness")
```

---

## Demo: Agente ReAct

```bash
export OPENROUTER_API_KEY=sk-or-...
cd py
python demo_react.py "What's the weather like in Madrid?"
python demo_react.py "3 * 7 + 12"
python demo_react.py "What is the capital of Spain?"
```

Demo funcional de un agente ReAct con 3 herramientas (weather, calculator, search) instrumentado con treval. Después de ejecutarlo:

```bash
treval spans         # Ver todos los spans generados
treval span 1        # Detalle del agente
treval eval          # Evaluar con LLM-as-judge
```

---

## Comandos (15)

| Comando | Descripción |
|---------|-------------|
| `treval spans` | Lista spans recientes (filtra por tipo) |
| `treval span <id>` | Detalle del span con hijos |
| `treval count` | Total de spans almacenados |
| `treval clear` | Borra todos los spans |
| `treval eval` | Evalúa spans con LLM-as-judge |
| `treval evals` | Historial de evaluaciones |
| `treval compare` | Compara N modelos × M ejecuciones |
| `treval ab` | Comparación A/B simple (legacy) |
| `treval replay <id>` | Re-ejecuta un span con nuevos parámetros |
| `treval test run <file>` | Ejecuta tests de agente |
| `treval dashboard` | Dashboard web / exportación HTML |
| `treval metrics` | Métricas agregadas |
| `treval prices` | Precios de API OpenRouter |
| `treval export` | Exporta spans a OTEL |
| `treval gateway` | Proxy para interceptar tráfico LLM |

---

## Almacenamiento

Todo se guarda localmente en `~/.treval/`:

```
~/.treval/
├── spans.db       # Trazas (spans con jerarquía padre→hijo)
└── evals.db       # Evaluaciones LLM-as-judge
```

SQLite, thread-safe, sin servidor. Puedes borrar los archivos en cualquier momento o usar `treval clear` (solo borra spans; las evaluaciones están en `evals.db` separado).

---

## Arquitectura

```
treval/
├── py/
│   ├── treval/
│   │   ├── __init__.py    # API pública (decoradores + instrument + eval)
│   │   ├── agent.py       # @agent — decorador para clases agente
│   │   ├── operation.py   # @operation — decorador para métodos
│   │   ├── tool.py        # @tool — decorador para funciones
│   │   ├── instrument.py  # Auto-instrumentación OpenAI síncrona/asíncrona
│   │   ├── wrap.py        # Wrappers para clientes existentes
│   │   ├── context.py     # Pila thread-local de span_id
│   │   ├── db.py          # SQLite local (~/.treval/spans.db)
│   │   ├── eval.py        # LLM-as-judge (parser JSON tolerante) + EvalStore
│   │   ├── compare.py     # Comparación multi-modelo + agente + informe HTML
│   │   ├── replay.py      # Re-ejecuta spans con parámetros modificados
│   │   ├── testing.py     # TestRunner nativo con @case y TestSuite
│   │   ├── callbacks.py   # Callbacks de trazado (compatibles con LangChain)
│   │   ├── otel.py        # Exportador OpenTelemetry
│   │   ├── gateway.py     # Proxy HTTP para interceptar tráfico LLM
│   │   ├── dashboard.py   # Dashboard web + exportación HTML autónoma
│   │   └── cli.py         # CLI con Rich (15 comandos)
│   ├── tests/             # 88 tests, todos pasando
│   └── demo_react.py      # Demo: agente ReAct funcional con 3 herramientas
├── ts/                    # Esqueleto TypeScript (futuro)
└── pyproject.toml         # Configuración del paquete
```

### Flujo de Datos

```
Llamada LLM
  │
  ├─ instrument() parchea OpenAI → span LLM guardado en SpanStore
  ├─ @agent / @operation / @tool → span AGENT/OPERATION/TOOL
  │
  ▼
SpanStore (SQLite) ─→ CLI (treval spans / span / metrics)
                  ─→ Dashboard (localhost:8080 o HTML autónomo)
                  ─→ LLM-as-judge → EvalStore
                  ─→ compare_models() → informe HTML con estadísticas y costes
                  ─→ Exportación OTEL (consola o collector)
                  ─→ Replay (re-ejecutar con nuevos parámetros)
```

---

## Tests

```bash
cd py
python -m pytest tests/ -v
```

**88 tests**, todos pasando. Desarrollados con TDD estricto: cada nueva funcionalidad empieza con un test RED, luego implementación GREEN, después refactor.

Cobertura: decoradores (`@agent`, `@operation`, `@tool`), auto-instrumentación, almacenamiento, evaluación, comparación (modelos + agente + precios API), replay, testing, generación HTML, parser JSON tolerante.

---

## Licencia

MIT