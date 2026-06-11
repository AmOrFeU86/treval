"""treval CLI: view, filter and explore spans from the terminal."""

import argparse
import sys
from pathlib import Path
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.text import Text

from treval.db import SpanStore
from treval.eval import EvalStore, LLMEvaluator, CORRECTNESS_EVALUATOR
from treval.otel import OtelExporter
from treval.testing import TestRunner, load_test_file
from treval.replay import ReplaySession, interactive_replay
from treval.wrap import wrap
from treval.dashboard import serve as dashboard_serve
from treval.dashboard import HTML_TEMPLATE as dashboard_html

console = Console()


def cmd_spans(args):
    """List the most recent spans in a table."""
    store = SpanStore()
    span_type = args.type
    limit = args.limit

    spans = store.list_spans(limit=limit, type=span_type)

    if not spans:
        console.print("[dim]No spans recorded.[/dim]")
        return

    table = Table(title=f"Recent spans{' (' + span_type + ')' if span_type else ''}")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Type", style="magenta")
    table.add_column("Name")
    table.add_column("Status", no_wrap=True)
    table.add_column("Duration", justify="right")
    table.add_column("Parent", style="dim")
    table.add_column("Start")

    for s in spans:
        status_style = "green" if s["status"] == "ok" else "red bold"
        duration = f"{s['duration_ms']:.1f}ms" if s["duration_ms"] is not None else "-"
        parent = str(s["parent_id"]) if s["parent_id"] else "—"
        created = s["created_at"][:19] if s["created_at"] else "-"

        table.add_row(
            str(s["id"]),
            s["type"],
            s["name"],
            Text(s["status"], style=status_style),
            duration,
            parent,
            created,
        )

    console.print(table)
    console.print(f"\n[dim]{len(spans)} spans • {store.count()} total[/dim]")


def cmd_span_detail(args):
    """Show details of a specific span + its children."""
    store = SpanStore()
    span = store.get(args.span_id)
    if not span:
        console.print(f"[red]Span {args.span_id} not found.[/red]")
        return

    console.print(f"\n[bold cyan]Span #{span['id']}[/bold cyan]")
    console.print(f"  [bold]Name:[/bold]        {span['name']}")
    console.print(f"  [bold]Type:[/bold]        {span['type']}")
    console.print(f"  [bold]Status:[/bold]      ", end="")
    if span["status"] == "ok":
        console.print("[green]ok[/green]")
    else:
        console.print(f"[red bold]{span['status']}[/red bold]")
    console.print(f"  [bold]Duration:[/bold]    {span['duration_ms']:.1f}ms" if span["duration_ms"] else "")
    console.print(f"  [bold]Parent:[/bold]      {span['parent_id'] or '—'}")
    console.print(f"  [bold]Start:[/bold]       {span['created_at']}")

    if span["input"]:
        console.print(f"\n  [bold]Input:[/bold]")
        console.print(f"    [dim]{span['input']}[/dim]")

    if span["output"]:
        console.print(f"\n  [bold]Output:[/bold]")
        console.print(f"    [dim]{span['output']}[/dim]")

    # Children
    children = store.get_children(args.span_id)
    if children:
        console.print(f"\n  [bold]Children ({len(children)}):[/bold]")
        for c in children:
            console.print(f"    #{c['id']} {c['type']} [bold]{c['name']}[/bold] [{c['status']}]"
                          f" ({c['duration_ms']:.1f}ms)" if c['duration_ms'] else "")


def cmd_count(args):
    """Show the total number of spans."""
    store = SpanStore()
    console.print(f"Total spans: [bold]{store.count()}[/bold]")


def cmd_clear(args):
    """Clear all spans."""
    confirm = input("Delete all spans? (y/N): ")
    if confirm.lower() == "y":
        store = SpanStore()
        store.clear()
        console.print("[green]Spans cleared.[/green]")
    else:
        console.print("[dim]Cancelled.[/dim]")


def cmd_eval(args):
    """Evaluate spans with LLM-as-judge."""
    store = SpanStore()
    eval_store = EvalStore()
    spans = store.list_spans(limit=args.limit, type=args.type)

    if not spans:
        console.print("[dim]No spans to evaluate.[/dim]")
        return

    # Choose evaluator based on criteria
    criteria_map = {
        "correctness": "The response is correct, accurate and factual",
        "conciseness": "The response is concise without irrelevant information",
        "helpfulness": "The response is useful and actionable",
        "groundedness": "The response uses the tools correctly",
    }
    criteria = criteria_map.get(args.criteria, args.criteria)
    evaluator = LLMEvaluator(name=args.criteria, criteria=criteria)

    console.print(f"[bold]Evaluating {len(spans)} spans with criteria:[/bold] {args.criteria}")
    console.print(f"[dim]Model: {evaluator.model}[/dim]\\n")

    results = evaluator.evaluate(spans)
    for r in results:
        eval_store.save(r)

    # Show results
    table = Table(title="Evaluation results")
    table.add_column("Span ID", style="cyan")
    table.add_column("Type")
    table.add_column("Score", justify="right")
    table.add_column("Reason")

    for r in results:
        score_color = "green" if r.score >= 0.7 else ("yellow" if r.score >= 0.4 else "red")
        table.add_row(
            str(r.span_id),
            r.metadata.get("span_type", ""),
            f"[{score_color}]{r.score:.2f}[/{score_color}]",
            r.reason[:80],
        )

    console.print(table)
    stats = eval_store.get_stats()
    console.print(f"\n[dim]{len(results)} evaluations • Avg: {stats['avg_score']:.2f}[/dim]")


def cmd_evals(args):
    """List evaluation results."""
    eval_store = EvalStore()
    results = eval_store.list(limit=20)

    if not results:
        console.print("[dim]No evaluations recorded.[/dim]")
        return

    table = Table(title="Recent evaluations")
    table.add_column("ID", style="cyan")
    table.add_column("Span", style="yellow")
    table.add_column("Evaluator")
    table.add_column("Score", justify="right")
    table.add_column("Reason")

    for r in results:
        score_color = "green" if r["score"] >= 0.7 else ("yellow" if r["score"] >= 0.4 else "red")
        table.add_row(
            str(r["id"]),
            f"#{r['span_id']} {(r.get('span_name') or '')[:30]}",
            r.get("evaluator_name", ""),
            f"[{score_color}]{r['score']:.2f}[/{score_color}]",
            (r.get("reason", "") or "")[:60],
        )

    console.print(table)
    stats = eval_store.get_stats()
    console.print(f"[dim]Total: {stats['count']} • Avg: {stats['avg_score']:.2f}[/dim]")

def cmd_export(args):
    """Export spans to OTEL format."""
    exporter = OtelExporter(
        endpoint=args.endpoint,
        use_console=args.console,
    )
    count = exporter.export_all(limit=args.limit)
    if args.console:
        console.print(f"[green]Exported {count} spans to OTEL console.[/green]")
    else:
        console.print(f"[green]Exported {count} spans to {exporter.endpoint}[/green]")


def cmd_test_run(args):
    """Run tests defined in a .py file."""
    try:
        suite = load_test_file(args.file)
    except (FileNotFoundError, ImportError, ValueError) as e:
        console.print(f"[red]Error loading test:[/red] {e}")
        return

    evaluator = LLMEvaluator(
        name="test",
        criteria="correctness",
    )
    runner = TestRunner(evaluator=evaluator)

    console.print(f"[bold]Running suite:[/bold] {suite.name}")
    console.print(f"[dim]{len(suite.tests)} tests[/dim]\n")

    results = runner.run_suite(suite)
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)

    for r in results:
        status = "[green]✅ PASSED[/green]" if r.passed else "[red]❌ FAILED[/red]"
        score_color = "green" if r.score >= 0.7 else "yellow" if r.score >= 0.4 else "red"
        console.print(f"{status} [bold]{r.name}[/bold]")
        console.print(f"   Input: [dim]{r.input[:80]}[/dim]")
        console.print(f"   Score: [{score_color}]{r.score:.2f}[/{score_color}] — {r.reason[:80]}")
        if r.error:
            console.print(f"   [red]Error: {r.error}[/red]")
        console.print()

    console.print(f"[bold]Result: {passed}/{len(results)} tests passed[/bold]")


def cmd_replay(args):
    """Re-run a span with optional parameter changes."""
    from treval.replay import ReplaySession
    try:
        session = ReplaySession(args.span_id)
    except ValueError as e:
        console.print(f"[red]❌ {e}[/red]")
        return

    console.print(f"[bold]🔄 Replay — #{args.span_id}[/bold] {session.span_type}: {session.span_name}")
    console.print(f"  Original input: [dim]{session.original.get('input', '')[:120]}[/dim]")

    if not session.is_replayable:
        console.print("[red]This span has no input to modify.[/red]")
        return

    if args.input:
        session.set_input(args.input)
    if args.model:
        session.set_model(args.model)
    if args.temperature is not None:
        session.set_temperature(args.temperature)

    console.print(f"  Using model: [bold]{session.modified_model or session._extract_model()}[/bold]")
    console.print()

    import os
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    result = session.replay(api_key)

    if "error" in result:
        console.print(f"[red]❌ {result['error']}[/red]")
        return

    console.print(f"[green]✅ Replay completed in {result['duration_ms']:.1f}ms[/green]")

    from rich.table import Table
    table = Table(title="Comparison")
    table.add_column("", style="bold")
    table.add_column("Original", style="dim")
    table.add_column("Replay")

    orig_out = (result.get("original_output", "") or "")[:200]
    new_out = (result.get("output", "") or "")[:200]
    orig_ms = f"{result.get('original_duration_ms', 0):.1f}ms"
    new_ms = f"{result.get('duration_ms', 0):.1f}ms"

    table.add_row("Output", orig_out, new_out)
    table.add_row("Duration", orig_ms, new_ms)

    usage = result.get("usage", {})
    if usage:
        table.add_row("Tokens",
                      "-",
                      f"P:{usage.get('prompt_tokens', '-')} C:{usage.get('completion_tokens', '-')} T:{usage.get('total_tokens', '-')}")

    console.print(table)


def cmd_gateway(args):
    """Start the proxy gateway for intercepting LLM traffic."""
    from treval.gateway import run_gateway
    run_gateway(host=args.host, port=args.port, upstream=args.upstream)


def cmd_dashboard(args):
    """Start the local web dashboard or export to HTML."""
    if args.export:
        from treval.dashboard import _build_html
        html = _build_html()
        Path(args.export).write_text(html, encoding="utf-8")
        spans_count = html.count("<tr")  # rough span count
        console.print(f"[green]✅ Dashboard exported to {args.export}[/green]")
        console.print(f"   {Path(args.export).stat().st_size:,} bytes")
        return
    dashboard_serve(port=args.port, open_browser=not args.no_open)


def cmd_metrics(args):
    """Show aggregated metrics of spans."""
    store = SpanStore()
    total = store.count()
    spans = store.list_spans(limit=1000)

    if not spans:
        console.print("[dim]No spans to show metrics for.[/dim]")
        return

    # Stats by type
    from collections import Counter, defaultdict
    types = Counter(s["type"] for s in spans)
    statuses = Counter(s["status"] for s in spans)

    durations = defaultdict(list)
    for s in spans:
        if s.get("duration_ms") is not None:
            durations[s["type"]].append(s["duration_ms"])

    # Main table
    t1 = Table(title="Span summary")
    t1.add_column("Metric", style="bold")
    t1.add_column("Value")
    t1.add_row("Total spans", str(total))
    t1.add_row("With errors", f"[red]{statuses.get('error', 0)}[/red]")
    t1.add_row("OK", f"[green]{statuses.get('ok', 0)}[/green]")
    console.print(t1)

    # Table by type
    t2 = Table(title="Metrics by type")
    t2.add_column("Type", style="bold")
    t2.add_column("Count")
    t2.add_column("Avg duration", justify="right")
    t2.add_column("Max duration", justify="right")
    t2.add_column("Error rate", justify="right")

    for span_type in ["AGENT", "OPERATION", "LLM", "TOOL"]:
        count = types.get(span_type, 0)
        if count == 0:
            continue
        avg_d = sum(durations.get(span_type, [])) / len(durations.get(span_type, [])) if durations.get(span_type) else 0
        max_d = max(durations.get(span_type, [0])) if durations.get(span_type) else 0
        err_rate = sum(1 for s in spans if s["type"] == span_type and s["status"] == "error") / count * 100

        t2.add_row(
            span_type,
            str(count),
            f"{avg_d:.0f}ms" if avg_d else "-",
            f"{max_d:.0f}ms" if max_d else "-",
            f"[red]{err_rate:.0f}%[/red]" if err_rate > 0 else "[green]0%[/green]",
        )

    console.print(t2)


def cmd_compare(args):
    """Compare N models on the same prompt, each M times, with stats and costs."""
    import os
    from treval.compare import compare_models, build_report_html, MODEL_PRICES
    from rich.table import Table

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        console.print("[red]❌ OPENROUTER_API_KEY not set[/red]")
        console.print("   Export it: export OPENROUTER_API_KEY=sk-...")
        return

    # Determine mode: direct prompt or agent
    if args.agent:
        from treval.compare import compare_agents
        agent_cmd = args.agent

        console.print(f"\n[bold]🧪 Agent mode:[/bold] {agent_cmd}")
        console.print(f"   Running {agent_cmd} × {args.runs} times")
        console.print()

        if not args.prompt:
            args.prompt = agent_cmd  # Use the command as prompt for evaluation

        with console.status("[bold green]Running agent...") as status:
            results = compare_agents(
                agent_cmd=agent_cmd,
                runs=args.runs,
                criteria=args.criteria,
                api_key=api_key,
            )
        prompt = agent_cmd
    else:
        prompt = args.prompt

    if not prompt:
        console.print("[red]❌ Need --prompt or --agent[/red]")
        return

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        console.print("[red]❌ No models specified[/red]")
        return

    if len(models) < 2:
        console.print("[yellow]⚠️ Only 1 model. Comparison needs at least 2.[/yellow]")

    console.print(f"\n[bold]📊 Comparing {len(models)} model(s) × {args.runs} run(s)[/bold]")
    console.print(f"   Prompt: [dim]{prompt[:120]}[/dim]")
    console.print(f"   Criteria: {args.criteria}")
    console.print(f"   Models: {', '.join(models)}")
    console.print()

    # Check prices via API
    with console.status("[dim]Getting prices from OpenRouter API...") as status:
        from treval.compare import get_model_price, list_available_models
        unknown = []
        for m in models:
            inp, out = get_model_price(m, api_key)
            source = "api"  # Can't easily know exact source here
            if inp == 5.0 and out == 15.0 and m not in getattr(list_available_models(api_key), "ids", []):
                unknown.append(m)
        if unknown:
            console.print(f"[dim]⚠️ {len(unknown)} model(s) without known price (using conservative estimate)[/dim]")

    with console.status("[bold green]Running comparison...") as status:
        results = compare_models(
            prompt=prompt,
            models=models,
            runs=args.runs,
            criteria=args.criteria,
            api_key=api_key,
        )

    # Show results
    best = max(results, key=lambda r: r.mean_score) if results else None

    if best:
        console.print(f"\n[bold green]🏆 Winner: {best.model}[/bold green] "
                      f"([green]{best.mean_score:.3f}[/green] ± {best.std_score:.3f})")

    table = Table(title="Model comparison", header_style="bold")
    table.add_column("#", style="dim")
    table.add_column("Model")
    table.add_column("Score", justify="right")
    table.add_column("±σ", justify="right")
    table.add_column("Duration", justify="right")
    table.add_column("Cost/run", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Runs", justify="right")

    for i, r in enumerate(results):
        is_winner = r.model == (best.model if best else None) and r.mean_score == (best.mean_score if best else 0)
        style = "bold green" if is_winner else ""
        score_str = f"{r.mean_score:.3f}"
        std_str = f"{r.std_score:.3f}" if r.run_count > 1 else "—"
        dur_str = f"{r.mean_duration:.0f}ms" if r.run_count > 0 else "—"
        from treval.compare import format_cost
        cost_str = format_cost(r.mean_cost) if r.run_count > 0 else "—"
        tokens_str = f"{r.total_tokens}" if r.run_count > 0 else "—"

        table.add_row(
            str(i + 1),
            f"{'🏆 ' if is_winner else ''}{r.model}",
            score_str, std_str, dur_str, cost_str, tokens_str,
            str(r.run_count),
            style=style,
        )

    console.print(table)

    # Resumen
    if results:
        total_cost = sum(r.total_cost for r in results)
        total_runs = sum(r.run_count for r in results)
        from treval.compare import format_cost
        console.print(f"\n[dim]Total: {total_runs} runs · Total cost: {format_cost(total_cost)}[/dim]")

    # Export to HTML
    if args.export:
        path = args.export
        with console.status("[yellow]Generating HTML report..."):
            html = build_report_html(results, prompt, args.criteria)
        import os as _os
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        abs_path = _os.path.abspath(path)
        file_size = _os.path.getsize(abs_path)
        console.print(f"[green]✅ Report exported to {abs_path}[/green]")
        console.print(f"   {file_size:,} bytes · {len(results)} model(s) · {total_runs} run(s)")
        console.print(f"   Open with: file://{abs_path}")


def cmd_init(args):
    """Create an agent project with treval preconfigured."""
    from pathlib import Path
    import shutil

    target = Path(args.path).expanduser().resolve()
    templates_dir = Path(__file__).parent / "templates"

    if target.exists() and any(target.iterdir()):
        console.print(f"[red]❌ {target} already exists and is not empty[/red]")
        return

    target.mkdir(parents=True, exist_ok=True)

    # Copy templates
    files = {
        "agent.py": "src/agent.py",
        "test_agent.py": "tests/test_agent.py",
    }
    for src_name, dest_rel in files.items():
        src = templates_dir / src_name
        if src.exists():
            dest = target / dest_rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            console.print(f"  [green]✅[/green] {dest_rel}")

    # Create .env
    env_path = target / ".env"
    if not env_path.exists():
        env_path.write_text("# Get your API key at https://openrouter.ai/keys\n")
        env_path.write_text("OPENROUTER_API_KEY=***")
        console.print(f"  [green]✅[/green] .env (edit it with your API key)")

    # Create project README.md
    readme = target / "README.md"
    if not readme.exists():
        readme.write_text("""# My agent with treval

Project generated with `treval init`.

## Usage

```bash
python src/agent.py "your question"
```

## View traces

```bash
treval spans
treval eval
treval dashboard --export dashboard.html
```

## Tests

```bash
treval test run tests/test_agent.py
```
""")
    console.print(f"\n[bold green]🎉 Project created in {target}[/bold green]")
    console.print(f"\n   [bold]Next steps:[/bold]")
    console.print("   1. Edit [bold].env[/bold] with your OPENROUTER_API_KEY")
    console.print(f"   2. [bold]cd {target}[/bold]")
    console.print("   3. [bold]python src/agent.py[/bold]")
    console.print("   4. [bold]treval spans[/bold] to view traces")
    console.print("   5. [bold]treval test run tests/test_agent.py[/bold]")

def cmd_prices(args):
    """Shows model prices from OpenRouter API."""
    from treval.compare import list_available_models, format_cost
    from rich.table import Table
    from rich.text import Text

    api_key = os.environ.get("OPENROUTER_API_KEY", "")

    with console.status("[dim]Getting prices from OpenRouter API...") as status:
        models = list_available_models(api_key)

    if not models:
        console.print("[red]❌ Could not fetch prices. No API connection and no fallback available.[/red]")
        return

    # Filter by search if provided
    if args.search:
        query = args.search.lower()
        models = [m for m in models if query in m["id"].lower()]

    source_label = "OpenRouter API" if models and models[0]["source"] == "api" else "fallback local"

    table = Table(
        title=f"Model prices ({source_label})",
        header_style="bold",
        show_lines=True,
    )
    table.add_column("Model", style="cyan")
    table.add_column("Input", justify="right")
    table.add_column("Output", justify="right")

    for m in models:
        inp_str = format_cost(m["input_price"] / 1_000_000) + "/tok"
        out_str = format_cost(m["output_price"] / 1_000_000) + "/tok"
        table.add_row(m["id"], inp_str, out_str)

    console.print(table)

    # Statistics
    source_api = sum(1 for m in models if m["source"] == "api")
    source_fb = sum(1 for m in models if m["source"] == "fallback")
    parts = []
    if source_api:
        parts.append(f"{source_api} from API")
    if source_fb:
        parts.append(f"{source_fb} from local fallback")
    console.print(f"\n[dim]{len(models)} models shown ({', '.join(parts)})[/dim]")
    if args.search:
        console.print(f"[dim]Search: {args.search}[/dim]")

    if args.limit and len(models) > args.limit:
        console.print(f"[yellow]⚠️ Showing {args.limit} of {len(models)} models. Use --search to filter.[/yellow]")


def cmd_ab(args):
    """A/B comparison: run 2 configs on the same input."""
    import os
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        console.print("[red]OPENROUTER_API_KEY not configured[/red]")
        return

    from treval.eval import LLMEvaluator
    from openai import OpenAI

    # Show what will be compared
    console.print(f"[bold]🧪 A/B Comparison[/bold]")
    console.print(f"  Input: [dim]{args.input[:100]}[/dim]")
    console.print(f"  A: {args.model_a} (temp={args.temp_a})")
    console.print(f"  B: {args.model_b} (temp={args.temp_b})")
    console.print()

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key,
                    default_headers={"HTTP-Referer": "https://treval.dev", "X-Title": "treval-ab"})
    evaluator = LLMEvaluator(name="ab", criteria="correctness",
                             model="deepseek/deepseek-v4-flash")

    results = []
    for label, model, temp in [("A", args.model_a, args.temp_a),
                                 ("B", args.model_b, args.temp_b)]:
        console.print(f"[bold]Running {label}...[/bold]")
        import time
        start = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": args.input}],
                temperature=temp,
                max_tokens=500,
            )
            output = response.choices[0].message.content or ""
            duration = (time.perf_counter() - start) * 1000
            usage = getattr(response, "usage", None)
            tokens = (usage.total_tokens if usage else 0) if hasattr(usage, "total_tokens") else 0
            score, reason = evaluator._llm_judge("correctness", args.input, output)
            results.append({
                "label": label,
                "model": model,
                "temp": temp,
                "output": output,
                "duration_ms": duration,
                "tokens": tokens,
                "score": score,
                "reason": reason,
            })
        except Exception as e:
            console.print(f"[red]Error en {label}: {e}[/red]")
            return

    # Show comparison
    from rich.table import Table
    table = Table(title="A/B Comparison")
    table.add_column("", style="bold")
    for r in results:
        table.add_column(f"{r['label']} — {r['model'].split('/')[-1][:20]}", style="bold")

    table.add_row("Output",
                  results[0]["output"][:300] if len(results) > 0 else "-",
                  results[1]["output"][:300] if len(results) > 1 else "-")
    table.add_row("Duration",
                  f"{results[0]['duration_ms']:.0f}ms" if len(results) > 0 else "-",
                  f"{results[1]['duration_ms']:.0f}ms" if len(results) > 1 else "-")
    table.add_row("Tokens",
                  str(results[0].get("tokens", 0)) if len(results) > 0 else "-",
                  str(results[1].get("tokens", 0)) if len(results) > 1 else "-")

    score_a = results[0].get("score", 0) if len(results) > 0 else 0
    score_b = results[1].get("score", 0) if len(results) > 1 else 0
    table.add_row("Score",
                  f"{score_a:.2f}" if len(results) > 0 else "-",
                  f"{score_b:.2f}" if len(results) > 1 else "-")

    console.print(table)
    if score_a > score_b:
        console.print(f"\n[green]🏆 Winner: A ({results[0]['model']}) with {score_a:.2f} vs {score_b:.2f}[/green]")
    elif score_b > score_a:
        console.print(f"\n[green]🏆 Winner: B ({results[1]['model']}) with {score_b:.2f} vs {score_a:.2f}[/green]")
    else:
        console.print("\n[dim]Technical tie[/dim]")


def main():
    parser = argparse.ArgumentParser(
        prog="treval",
        description="Trace, evaluate and improve AI agents from the terminal.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # treval spans
    p_spans = sub.add_parser("spans", help="List recent spans")
    p_spans.add_argument("-t", "--type", help="Filter by type (TOOL, AGENT, OPERATION)")
    p_spans.add_argument("-l", "--limit", type=int, default=20, help="Number of spans (default: 20)")
    p_spans.set_defaults(func=cmd_spans)

    # treval span <id>
    p_span = sub.add_parser("span", help="Detail of a span")
    p_span.add_argument("span_id", type=int, help="Span ID")
    p_span.set_defaults(func=cmd_span_detail)

    # treval count
    sub.add_parser("count", help="Total number of spans").set_defaults(func=cmd_count)

    # treval clear
    sub.add_parser("clear", help="Delete all spans").set_defaults(func=cmd_clear)

    # treval eval
    p_eval = sub.add_parser("eval", help="Evaluate spans with LLM-as-judge")
    p_eval.add_argument("-c", "--criteria", default="correctness",
                        help="Evaluation criteria (correctness, conciseness, helpfulness)")
    p_eval.add_argument("-t", "--type", help="Span type to evaluate (TOOL, OPERATION, LLM)")
    p_eval.add_argument("-l", "--limit", type=int, default=10)
    p_eval.set_defaults(func=cmd_eval)

    # treval evals
    sub.add_parser("evals", help="List evaluation results").set_defaults(func=cmd_evals)

    # treval export otlp
    p_export = sub.add_parser("export", help="Export spans to OTEL/OTLP")
    p_export.add_argument("--endpoint", default=None,
                          help="OTLP endpoint (e.g., http://localhost:4317)")
    p_export.add_argument("--console", action="store_true",
                          help="Export to console (stdout)")
    p_export.add_argument("-l", "--limit", type=int, default=1000)
    p_export.set_defaults(func=cmd_export)

    # treval test run
    p_test = sub.add_parser("test", help="Run tests for agents")
    p_test_sub = p_test.add_subparsers(dest="test_subcommand", required=True)
    p_test_run = p_test_sub.add_parser("run", help="Run a test file")
    p_test_run.add_argument("file", help=".py file with the test suite")
    p_test_run.set_defaults(func=cmd_test_run)

    # treval replay <span_id>
    p_replay = sub.add_parser("replay", help="Re-execute a span with modified params")
    p_replay.add_argument("span_id", type=int, help="Span ID to re-execute")
    p_replay.add_argument("--input", help="New input (optional)")
    p_replay.add_argument("--model", help="Model to use (default: original)")
    p_replay.add_argument("--temperature", type=float, default=None, help="Temperature (default: 0.1)")
    p_replay.set_defaults(func=cmd_replay)

    # treval dashboard
    p_dash = sub.add_parser("dashboard", help="Start the local web dashboard")
    p_dash.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    p_dash.add_argument("--no-open", action="store_true", help="Don't open browser")
    p_dash.add_argument("--export", type=str, default=None,
                        help="Export to static HTML file (e.g., dashboard.html)")
    p_dash.set_defaults(func=cmd_dashboard)

    # treval metrics
    sub.add_parser("metrics", help="Show aggregated span metrics").set_defaults(func=cmd_metrics)

    # treval compare
    p_compare = sub.add_parser("compare", help="Compare N models × M runs with stats and costs")
    p_compare.add_argument("-p", "--prompt", help="Prompt for comparison")
    p_compare.add_argument("-m", "--models", default="deepseek/deepseek-v4-flash,deepseek/deepseek-v4-pro",
                           help="Models separated by commas (default: flash,pro)")
    p_compare.add_argument("-r", "--runs", type=int, default=3,
                           help="Number of runs per model (default: 3)")
    p_compare.add_argument("-c", "--criteria", default="correctness",
                           help="Evaluation criteria (correctness, conciseness, helpfulness)")
    p_compare.add_argument("-o", "--export", help="Export HTML report to file (e.g., compare.html)")
    p_compare.add_argument("-a", "--agent", help="Agent command to compare (agent mode)")
    p_compare.set_defaults(func=cmd_compare)

    # treval prices
    p_prices = sub.add_parser("prices", help="Show model prices from OpenRouter API")
    p_prices.add_argument("-s", "--search", help="Filter models by name")
    p_prices.add_argument("-l", "--limit", type=int, default=0, help="Limit number of results")
    p_prices.set_defaults(func=cmd_prices)

    # treval init
    p_init = sub.add_parser("init", help="Create an agent project with treval preconfigured")
    p_init.add_argument("path", nargs="?", default="./mi-agente",
                        help="Project path (default: ./mi-agente)")
    p_init.set_defaults(func=cmd_init)

    # treval ab
    p_ab = sub.add_parser("ab", help="A/B comparison between 2 models/configs")
    p_ab.add_argument("input", help="Input for comparison")
    p_ab.add_argument("--model-a", default="deepseek/deepseek-v4-flash", help="Model A (default: flash)")
    p_ab.add_argument("--model-b", default="deepseek/deepseek-v4-pro", help="Model B (default: pro)")
    p_ab.add_argument("--temp-a", type=float, default=0.1, help="Temperature A (default: 0.1)")
    p_ab.add_argument("--temp-b", type=float, default=0.7, help="Temperature B (default: 0.7)")
    p_ab.set_defaults(func=cmd_ab)

    # treval gateway
    p_gw = sub.add_parser("gateway", help="Start the gateway proxy to intercept LLM traffic")
    p_gw.add_argument("--port", type=int, default=9090, help="Port (default: 9090)")
    p_gw.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1)")
    p_gw.add_argument("--upstream", choices=["openrouter", "openai"], default="openrouter",
                      help="API upstream (default: openrouter)")
    p_gw.set_defaults(func=cmd_gateway)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()