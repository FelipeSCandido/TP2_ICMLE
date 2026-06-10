"""
report_generator.py — Componente 4 do TP2
Gera relatórios de inspeção em Markdown com contexto histórico do RAG
e integração opcional com dados de trajectória do Projecto 1.
"""

import os
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()
# Garante que encontra o .env na raiz do projecto mesmo executando de src/
_root = Path(__file__).resolve().parent.parent
if (_root / ".env").exists():
    load_dotenv(_root / ".env", override=True)

P1_METRICS_JSON = os.getenv("P1_METRICS_JSON", "")
P1_INSIGHTS_JSON = os.getenv("P1_INSIGHTS_JSON", "")


def _load_p1_data() -> tuple[Optional[dict], Optional[dict]]:
    """Carrega dados do Projecto 1 se disponíveis."""
    metrics = None
    insights = None

    if P1_METRICS_JSON and Path(P1_METRICS_JSON).exists():
        with open(P1_METRICS_JSON) as f:
            metrics = json.load(f)

    if P1_INSIGHTS_JSON and Path(P1_INSIGHTS_JSON).exists():
        with open(P1_INSIGHTS_JSON) as f:
            insights = json.load(f)

    return metrics, insights


def _format_status_badge(status: str) -> str:
    badges = {
        "ok": "🟢 OK",
        "warning": "🟡 AVISO",
        "critical": "🔴 CRÍTICO",
    }
    return badges.get(status, status.upper())


def _format_severity_badge(severity: str) -> str:
    badges = {
        "low": "⬇ baixa",
        "medium": "➡ média",
        "high": "⬆ alta",
    }
    return badges.get(severity, severity)


def _get_zone_p1_context(zone_id: str, metrics: Optional[dict]) -> Optional[str]:
    """Extrai contexto de afluência do Projecto 1 para uma zona."""
    if not metrics:
        return None

    zone_data = metrics.get("zone_metrics", {}).get(zone_id)
    if not zone_data:
        return None

    total = zone_data.get("total_entries", 0)
    avg_dwell = zone_data.get("avg_dwell_s", 0)
    stop_rate = zone_data.get("stop_rate", 0)

    lines = [
        f"- **Visitas na semana:** {total:,}",
        f"- **Dwell médio:** {avg_dwell:.0f}s ({avg_dwell/60:.1f} min)",
        f"- **Stop rate:** {stop_rate:.1%}",
    ]

    # Anomalias na zona
    anomalies = [a for a in metrics.get("anomalies", []) if a.get("zone") == zone_id]
    if anomalies:
        lines.append(f"- **Anomalias detectadas (Proj.1):** {len(anomalies)}")
        for a in anomalies[:3]:
            lines.append(
                f"  - Às {a['hour']}h em {a['date']}: {a['actual']} visitantes "
                f"(esperado {a['expected']:.1f}, {a['direction']})"
            )

    return "\n".join(lines)


def generate_inspection_report(
    inspections: list[dict],
    session_name: str = "hoje",
    include_p1: bool = True,
    notifications: Optional[list[dict]] = None,
    rag_results: Optional[list[dict]] = None,
    output_path: Optional[str] = None,
) -> str:
    """
    Gera relatório Markdown para uma sessão de inspeção.

    Args:
        inspections: Lista de dicionários de inspeção (output do shelf_inspector)
        session_name: Nome da sessão (para cabeçalho)
        include_p1: Se True, integra dados do Projecto 1
        notifications: Lista de notificações do rule_engine
        rag_results: Resultados RAG relevantes para contexto histórico
        output_path: Caminho para guardar o ficheiro .md (opcional)

    Returns:
        String com o relatório em Markdown
    """
    now = datetime.now()
    metrics, p1_insights = _load_p1_data() if include_p1 else (None, None)

    # ── Estatísticas globais da sessão ─────────────────────────────────────
    n_zones = len(set(i.get("zone_id") for i in inspections))
    n_critical = sum(1 for i in inspections if i.get("overall_status") == "critical")
    n_warning = sum(1 for i in inspections if i.get("overall_status") == "warning")
    n_ok = sum(1 for i in inspections if i.get("overall_status") == "ok")
    all_issues = [iss for i in inspections for iss in i.get("issues", [])]
    n_issues_high = sum(1 for iss in all_issues if iss.get("severity") == "high")
    avg_fill = sum(i.get("shelf_fill_rate", 0) for i in inspections) / max(len(inspections), 1)
    notifications = notifications or []

    # ── Sumário executivo ──────────────────────────────────────────────────
    exec_summary_lines = [
        f"Sessão **{session_name}** com **{len(inspections)} zona(s) inspeccionada(s)** "
        f"({n_zones} distintas).",
        f"**{n_critical} críticas**, **{n_warning} com avisos**, **{n_ok} sem problemas**.",
        f"Fill rate médio: **{avg_fill:.0%}**. Issues de alta severidade: **{n_issues_high}**.",
    ]
    if notifications:
        crit_notifs = sum(1 for n in notifications if n.get("alert_level") == "critical")
        exec_summary_lines.append(
            f"**{len(notifications)} regras disparadas** ({crit_notifs} críticas)."
        )

    lines = [
        f"# Relatório de Inspeção de Prateleiras",
        f"",
        f"> Gerado automaticamente em {now.strftime('%d/%m/%Y às %H:%M')}",
        f"> Sessão: **{session_name}**",
        f"",
        f"---",
        f"",
        f"## 1. Sumário Executivo",
        f"",
        *[f"{l}" for l in exec_summary_lines],
        f"",
        f"---",
        f"",
        f"## 2. Problemas por Zona",
        f"",
    ]

    # ── Problemas por zona ──────────────────────────────────────────────────
    for insp in inspections:
        zone = insp.get("zone_id", "?")
        status = insp.get("overall_status", "ok")
        fill = insp.get("shelf_fill_rate", 1.0)
        issues = insp.get("issues", [])
        insp_id = insp.get("inspection_id", "?")
        strategy = insp.get("_strategy", "?")

        lines += [
            f"### {zone} — {_format_status_badge(status)}",
            f"",
            f"- **Inspeção:** `{insp_id}` (estratégia: {strategy})",
            f"- **Fill rate:** {fill:.0%}",
            f"- **Produtos detectados:** {', '.join(insp.get('products_detected', ['—']))}",
            f"",
        ]

        if issues:
            lines.append("**Issues detectados:**")
            lines.append("")
            for iss in issues:
                lines += [
                    f"- [{_format_severity_badge(iss.get('severity','low'))}] "
                    f"**{iss.get('type','?')}** @ {iss.get('location','?')}",
                    f"  - {iss.get('description','')}",
                    f"  - Confiança: {iss.get('confidence',0):.0%} | "
                    f"Área afectada: {iss.get('affected_area_pct',0):.0%}",
                    f"",
                ]
        else:
            lines.append("_Sem issues detectados._")
            lines.append("")

        # Contexto histórico do RAG para esta zona
        if rag_results:
            zone_rag = [r for r in rag_results if r.get("zone_id") == zone]
            if zone_rag:
                lines.append("**Histórico (RAG):**")
                for r in zone_rag[:2]:
                    lines.append(
                        f"- `{r.get('inspection_id')}` ({r.get('timestamp','')}): "
                        f"{r.get('summary','')[:100]}..."
                    )
                lines.append("")

        # Contexto Projecto 1
        if include_p1 and metrics:
            p1_ctx = _get_zone_p1_context(zone, metrics)
            if p1_ctx:
                lines += [
                    "**Afluência (Projecto 1):**",
                    "",
                    p1_ctx,
                    "",
                ]

        lines.append("---")
        lines.append("")

    # ── Regras disparadas ──────────────────────────────────────────────────
    lines += [
        "## 3. Regras Disparadas",
        "",
    ]
    if notifications:
        for notif in notifications:
            level_icon = {"info": "ℹ", "warning": "⚠️", "critical": "🚨"}.get(
                notif.get("alert_level", "info"), "ℹ"
            )
            lines += [
                f"### {level_icon} {notif.get('rule_id')} — {notif.get('rule_description','')}",
                f"",
                f"- **Nível:** {notif.get('alert_level','').upper()}",
                f"- **Inspeção:** `{notif.get('inspection_id','')}`",
                f"- **Zona:** {notif.get('zone_id','')}",
                f"- **Mensagem:** {notif.get('message','')}",
                f"",
            ]
    else:
        lines += ["_Nenhuma regra disparou nesta sessão._", ""]

    # ── Contexto histórico do RAG ──────────────────────────────────────────
    lines += [
        "## 4. Contexto Histórico (RAG)",
        "",
    ]
    if rag_results:
        for r in rag_results[:5]:
            lines += [
                f"- **`{r.get('inspection_id')}`** ({r.get('zone_id')}, "
                f"{r.get('timestamp','')}, sim={r.get('similarity',0):.2f})",
                f"  {r.get('summary','')[:150]}",
                f"",
            ]
    else:
        lines += ["_Sem histórico relevante recuperado._", ""]

    # ── Recomendações ──────────────────────────────────────────────────────
    lines += [
        "## 5. Recomendações",
        "",
        "_Ordenadas por urgência:_",
        "",
    ]

    # Reúne todas as recomendações a partir dos issues de alta severidade
    recommendations = []
    for insp in inspections:
        zone = insp.get("zone_id", "?")
        for iss in insp.get("issues", []):
            sev = iss.get("severity", "low")
            urgency = {"high": 0, "medium": 1, "low": 2}.get(sev, 2)
            recommendations.append({
                "urgency": urgency,
                "severity": sev,
                "zone": zone,
                "type": iss.get("type", ""),
                "location": iss.get("location", ""),
                "description": iss.get("description", ""),
            })

    # Adiciona notificações críticas
    for notif in notifications:
        if notif.get("alert_level") == "critical":
            recommendations.insert(0, {
                "urgency": -1,
                "severity": "high",
                "zone": notif.get("zone_id", "?"),
                "type": "rule_alert",
                "location": "—",
                "description": notif.get("message", ""),
            })

    recommendations.sort(key=lambda x: x["urgency"])

    for i, rec in enumerate(recommendations[:5], 1):
        sev_badge = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(rec["severity"], "")
        lines.append(
            f"{i}. {sev_badge} **[{rec['zone']}]** {rec['type']} @ {rec['location']}"
        )
        lines.append(f"   {rec['description']}")
        lines.append("")

    if not recommendations:
        lines += ["_Sem recomendações urgentes._", ""]

    # ── Integração com Projecto 1 ──────────────────────────────────────────
    if include_p1 and (metrics or p1_insights):
        lines += [
            "## 6. Integração com Projecto 1",
            "",
        ]
        if metrics:
            traffic = metrics.get("traffic", {})
            lines += [
                "**Visão geral da semana (dados de trajectória):**",
                "",
                f"- Total visitantes semana: **{traffic.get('total_visitors_week', 0):,}**",
                f"- Dia mais movimentado: **{traffic.get('busiest_day', '?')}**",
                f"- Hora de pico: **{traffic.get('peak_hour', '?')}h**",
                f"- Duração média de visita: **{traffic.get('avg_visit_duration_min', 0):.1f} min**",
                "",
            ]

        if p1_insights:
            selected_insights = p1_insights.get("insights", [])[:3]
            if selected_insights:
                lines.append("**Insights do Projecto 1 relevantes:**")
                lines.append("")
                for ins in selected_insights:
                    lines += [
                        f"- **{ins.get('titulo','')}**",
                        f"  {ins.get('implicacao','')}",
                        f"  *Recomendação: {ins.get('recomendacao','')}*",
                        "",
                    ]

        # Correlação entre issues visuais e anomalias de tráfego
        if metrics:
            anomalies = metrics.get("anomalies", [])
            zones_with_issues = {i.get("zone_id") for i in inspections if i.get("issues")}
            correlated = [a for a in anomalies if a.get("zone") in zones_with_issues]
            if correlated:
                lines += [
                    "**Correlação: zonas com issues visuais e anomalias de afluência:**",
                    "",
                ]
                for a in correlated[:3]:
                    lines.append(
                        f"- **{a['zone']}** às {a['hour']}h: {a['actual']} visitantes "
                        f"({a['direction']} do esperado {a['expected']:.1f}) — "
                        f"pode justificar issues de prateleira."
                    )
                lines.append("")

    # ── Footer ──────────────────────────────────────────────────────────────
    lines += [
        "---",
        "",
        "*Relatório gerado automaticamente pelo sistema de Retail Vision Intelligence (TP2).*",
    ]

    report_md = "\n".join(lines)

    # Guarda se pedido
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_md)
        print(f"[OK] Relatório guardado em: {output_path}")

    return report_md


def generate_zone_report(
    zone_id: str,
    period_days: int = 14,
    inspections_dir: Optional[str] = None,
) -> str:
    """
    Gera relatório focado numa zona específica para um período.
    """
    from rag_memory import query

    dir_path = Path(inspections_dir or os.getenv("INSPECTIONS_DIR", "./data/inspections"))
    all_files = list(dir_path.glob("INS_*.json"))

    zone_inspections = []
    for f in all_files:
        try:
            with open(f) as fp:
                insp = json.load(fp)
            if insp.get("zone_id") == zone_id:
                zone_inspections.append(insp)
        except Exception:
            pass

    # Query RAG para contexto histórico
    rag = query(f"problemas históricos na zona {zone_id}", k=5, zone_filter=zone_id)

    return generate_inspection_report(
        inspections=zone_inspections,
        session_name=f"Zona {zone_id} — últimos {period_days} dias",
        rag_results=rag.get("sources", []),
    )


# --- CLI simples ---
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso:")
        print("  python report_generator.py session <ficheiro_inspecao.json> [output.md]")
        print("  python report_generator.py zone <ZONE_ID>")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "session" and len(sys.argv) > 2:
        with open(sys.argv[2]) as f:
            insp = json.load(f)
        if not isinstance(insp, list):
            insp = [insp]
        out = sys.argv[3] if len(sys.argv) > 3 else None
        report = generate_inspection_report(insp, output_path=out)
        print(report)
    elif cmd == "zone" and len(sys.argv) > 2:
        report = generate_zone_report(sys.argv[2])
        print(report)
    else:
        print(f"Comando desconhecido: {cmd}")
