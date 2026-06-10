import os
import sys
import json
import shlex
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

load_dotenv()
_root = Path(__file__).resolve().parent.parent
if (_root / ".env").exists():
    load_dotenv(_root / ".env", override=True)

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.markdown import Markdown
    from rich.prompt import Prompt
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class RetailIntelligenceInterface:
    """Interface CLI para o sistema de Retail Vision Intelligence."""

    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None
        self.session_inspections = []
        self.session_notifications = []

    def _print(self, msg: str, style: str = ""):
        if self.console:
            self.console.print(msg, style=style)
        else:
            print(msg)

    def _print_panel(self, content: str, title: str = "", border_style: str = "blue"):
        if RICH_AVAILABLE and self.console:
            self.console.print(Panel(content, title=title, border_style=border_style))
        else:
            print(f"\n{'='*50}")
            if title:
                print(f"  {title}")
                print(f"{'='*50}")
            print(content)
            print('='*50)

    def _print_markdown(self, md: str):
        if RICH_AVAILABLE and self.console:
            self.console.print(Markdown(md))
        else:
            print(md)

    def cmd_inspect(self, args: list[str]):
        from shelf_inspector import inspect_image, inspect_batch
        from rule_engine import execute_rules
        from rag_memory import index_inspection

        if not args:
            self._print("[ERRO] Uso: inspect <ZONE_ID> --image <caminho>", "red")
            return

        zone = args[0]

        if zone == "all":
            images_dir = None
            for i, a in enumerate(args):
                if a == "--images-dir" and i + 1 < len(args):
                    images_dir = args[i + 1]

            if not images_dir:
                self._print("[ERRO] Falta --images-dir <directoria>", "red")
                return

            self._print(f"\n[Inspecção em batch] Directoria: {images_dir}", "cyan")
            results = inspect_batch(images_dir, zone_id="Z_S1")
            for r in results:
                self.session_inspections.append(r)
                notifs = execute_rules(r)
                self.session_notifications.extend(notifs)
                index_inspection(r)
            self._print(f"[OK] {len(results)} imagens inspeccionadas.", "green")

        else:
            image_path = None
            strategy = "cot"
            for i, a in enumerate(args):
                if a == "--image" and i + 1 < len(args):
                    image_path = args[i + 1]
                if a == "--strategy" and i + 1 < len(args):
                    strategy = args[i + 1]

            if not image_path:
                self._print("[ERRO] Falta --image <caminho>", "red")
                return

            self._print(f"\n[Inspecção] Zona: {zone} | Imagem: {image_path} | Estratégia: {strategy}", "cyan")

            try:
                result = inspect_image(image_path, zone_id=zone, strategy=strategy)
                self.session_inspections.append(result)

                notifs = execute_rules(result)
                self.session_notifications.extend(notifs)

                index_inspection(result)

                status = result.get("overall_status", "ok")
                fill = result.get("shelf_fill_rate", 1.0)
                n_issues = len(result.get("issues", []))
                colors = {"ok": "green", "warning": "yellow", "critical": "red"}
                color = colors.get(status, "white")

                self._print(f"\nStatus: {status.upper()} | Fill rate: {fill:.0%} | Issues: {n_issues}", color)

                for iss in result.get("issues", []):
                    self._print(
                        f"  [{iss.get('severity','?').upper()}] {iss.get('type')} @ {iss.get('location')}: {iss.get('description','')}",
                        "yellow"
                    )

                if notifs:
                    self._print(f"\n⚠️  {len(notifs)} regra(s) disparada(s):", "red")
                    for n in notifs:
                        self._print(f"  [{n['alert_level'].upper()}] {n['message']}", "red")

            except FileNotFoundError:
                self._print(f"[ERRO] Imagem não encontrada: {image_path}", "red")
            except ValueError as e:
                self._print(f"[ERRO] {e}", "red")
            except RuntimeError as e:
                self._print(f"[QUOTA] {e}", "yellow")

    def cmd_add_rule(self, args: list[str]):
        from rule_engine import add_rule

        if not args:
            self._print("[ERRO] Falta o texto da regra.", "red")
            return

        rule_text = " ".join(args)
        self._print(f"\n[Rule Engine] Adicionando regra: {rule_text}", "cyan")

        try:
            rule = add_rule(rule_text, interactive=True)
            self._print(f"\n[OK] Regra {rule.get('rule_id')} adicionada.", "green")
            if rule.get("validation", {}).get("ambiguities"):
                self._print("Ambiguidades registadas na regra. Reformula se necessário.", "yellow")
        except Exception as e:
            self._print(f"[ERRO] {e}", "red")

    def cmd_list_rules(self, args: list[str]):
        from rule_engine import list_rules

        rules = list_rules()
        if not rules:
            return

        if RICH_AVAILABLE and self.console:
            table = Table(title="Regras Activas")
            table.add_column("ID", style="cyan")
            table.add_column("Descrição")
            table.add_column("Alert", style="yellow")
            table.add_column("Zonas")
            for r in rules:
                conds = r.get("conditions", {})
                table.add_row(
                    r.get("rule_id", "?"),
                    r.get("description", r.get("natural_language", ""))[:60],
                    r.get("action", {}).get("alert_level", "?"),
                    str(conds.get("zone_filter") or "todas"),
                )
            self.console.print(table)

    def cmd_delete_rule(self, args: list[str]):
        from rule_engine import delete_rule

        if not args:
            self._print("[ERRO] Falta o ID da regra.", "red")
            return
        delete_rule(args[0])

    def cmd_test_rule(self, args: list[str]):
        from rule_engine import test_rule
        from shelf_inspector import inspect_image

        if len(args) < 3 or args[1] != "--image":
            self._print("[ERRO] Uso: test rule <RULE_ID> --image <caminho>", "red")
            return

        rule_id = args[0]
        image_path = args[2]

        try:
            inspection = inspect_image(image_path, strategy="cot")
            result = test_rule(rule_id, inspection)
            if result["fired"]:
                self._print(f"[DISPARO] Regra {rule_id} dispararia!", "red")
                for iss in result["triggered_issues"]:
                    self._print(f"  - {iss}", "yellow")
            else:
                self._print(f"[OK] Regra {rule_id} não dispararia.", "green")
        except Exception as e:
            self._print(f"[ERRO] {e}", "red")

    def cmd_history(self, args: list[str]):
        from rag_memory import query

        if not args:
            self._print("[ERRO] Falta a pergunta.", "red")
            return

        zone = None
        question_parts = []
        i = 0
        while i < len(args):
            if args[i] == "--zone" and i + 1 < len(args):
                zone = args[i + 1]
                i += 2
            else:
                question_parts.append(args[i])
                i += 1

        question = " ".join(question_parts)
        self._print(f"\n[RAG] Consultando histórico: {question}", "cyan")

        try:
            result = query(question, k=3, zone_filter=zone)
            self._print_panel(result["answer"], title="Resposta", border_style="green")
            self._print(f"\nFontes ({len(result['sources'])}):", "dim")
            for s in result["sources"]:
                self._print(
                    f"  - {s.get('inspection_id')} (zona {s.get('zone_id')}, "
                    f"sim={s.get('similarity',0):.2f})",
                    "dim"
                )
        except Exception as e:
            self._print(f"[ERRO] {e}", "red")

    def cmd_compare(self, args: list[str]):
        from rag_memory import query

        if len(args) < 2:
            self._print("[ERRO] Uso: compare <ZONE_A> <ZONE_B>", "red")
            return

        zone_a, zone_b = args[0], args[1]
        self._print(f"\n[Comparação] {zone_a} vs {zone_b}", "cyan")

        for zone in [zone_a, zone_b]:
            result = query(f"histórico de problemas na zona {zone}", k=3, zone_filter=zone)
            self._print(f"\n**{zone}:**")
            self._print(result["answer"])

    def cmd_report(self, args: list[str]):
        from report_generator import generate_inspection_report, generate_zone_report
        from rag_memory import query

        if not self.session_inspections and "--session" in args:
            self._print("[AVISO] Sem inspeções nesta sessão. Usa 'inspect' primeiro.", "yellow")
            return

        output_path = None
        for i, a in enumerate(args):
            if a == "--output" and i + 1 < len(args):
                output_path = args[i + 1]

        if "--zone" in args:
            zone_idx = args.index("--zone")
            zone_id = args[zone_idx + 1] if zone_idx + 1 < len(args) else None
            if not zone_id:
                self._print("[ERRO] Falta o ID da zona.", "red")
                return
            period = 14
            for i, a in enumerate(args):
                if a == "--period" and i + 1 < len(args):
                    import re
                    m = re.search(r'\d+', args[i + 1])
                    if m:
                        period = int(m.group())
            report = generate_zone_report(zone_id, period_days=period)
        else:
            rag = query("issues e anomalias recentes", k=5)
            report = generate_inspection_report(
                inspections=self.session_inspections,
                session_name="sessão actual",
                notifications=self.session_notifications,
                rag_results=rag.get("sources", []),
                output_path=output_path,
            )

        self._print_markdown(report)

    def cmd_rag_index(self, args: list[str]):
        from rag_memory import index_all_inspections

        d = None
        for i, a in enumerate(args):
            if a == "--dir" and i + 1 < len(args):
                d = args[i + 1]

        count = index_all_inspections(d)
        self._print(f"[OK] {count} inspeções indexadas no RAG.", "green")

    def _show_help(self):
        help_text = """
# Comandos Disponíveis

## Inspecção
  inspect <ZONE_ID> --image <caminho> [--strategy zero_shot|cot|few_shot]
  inspect all --images-dir <directoria>

## Regras
  add rule \"<regra em português>\"
  list rules
  delete rule <RULE_ID>
  test rule <RULE_ID> --image <caminho>

## Histórico (RAG)
  history \"<pergunta>\" [--zone <ZONE_ID>]
  compare <ZONE_A> <ZONE_B>
  rag index [--dir <directoria>]

## Relatórios
  report --session today [--output <ficheiro.md>]
  report --zone <ZONE_ID> [--period \"last 14 days\"]

## Outros
  help
  exit / quit
"""
        self._print_markdown(help_text)

    def run(self):
        """Loop principal da interface CLI."""
        self._print_panel(
            "Retail Vision Intelligence System — TP2\n"
            "Escreve 'help' para ver os comandos disponíveis.",
            title="Bem-vindo",
            border_style="blue"
        )

        while True:
            try:
                if RICH_AVAILABLE:
                    user_input = Prompt.ask("\n[bold blue]>[/bold blue]").strip()
                else:
                    user_input = input("\n> ").strip()

                if not user_input:
                    continue

                parts = shlex.split(user_input)
                if not parts:
                    continue
                cmd = parts[0].lower()
                args = parts[1:]

                if cmd in ("exit", "quit", "sair"):
                    self._print("Até logo!", "blue")
                    break
                elif cmd == "help":
                    self._show_help()
                elif cmd == "inspect":
                    self.cmd_inspect(args)
                elif cmd == "add" and args and args[0] == "rule":
                    self.cmd_add_rule(args[1:])
                elif cmd == "list" and args and args[0] == "rules":
                    self.cmd_list_rules(args[1:])
                elif cmd == "delete" and args and args[0] == "rule":
                    self.cmd_delete_rule(args[1:])
                elif cmd == "test" and args and args[0] == "rule":
                    self.cmd_test_rule(args[1:])
                elif cmd == "history":
                    self.cmd_history(args)
                elif cmd == "compare":
                    self.cmd_compare(args)
                elif cmd == "report":
                    self.cmd_report(args)
                elif cmd == "rag" and args and args[0] == "index":
                    self.cmd_rag_index(args[1:])
                else:
                    self._print(
                        f"[AVISO] Comando desconhecido: '{user_input}'. Escreve 'help' para ajuda.",
                        "yellow"
                    )

            except KeyboardInterrupt:
                self._print("\nInterrompido. Escreve 'exit' para sair.", "yellow")
            except EOFError:
                break
            except Exception as e:
                self._print(f"[ERRO inesperado] {type(e).__name__}: {e}", "red")


def main():
    app = RetailIntelligenceInterface()
    app.run()


if __name__ == "__main__":
    main()