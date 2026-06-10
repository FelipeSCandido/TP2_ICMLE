import os
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()
_root = Path(__file__).resolve().parent.parent
if (_root / ".env").exists():
    load_dotenv(_root / ".env", override=True)

RULES_DIR = Path(os.getenv("RULES_DIR", "./data/rules"))
RULES_DIR.mkdir(parents=True, exist_ok=True)

RULE_CONVERSION_PROMPT = """Você é um sistema de conversão de regras de inspeção de prateleiras de supermercado.
Converte a regra em linguagem natural para o schema JSON exacto abaixo.

REGRA DO GESTOR: "{RULE_TEXT}"

Converte para este schema JSON (responde APENAS com JSON válido, sem texto extra):
{{
  "rule_id": "{RULE_ID}",
  "created_at": "{TIMESTAMP}",
  "natural_language": "{RULE_TEXT}",
  "description": "reformulação clara e formal da regra em português",
  "conditions": {{
    "zone_filter": ["Z_S1"],
    "time_filter": {{"hours_start": null, "hours_end": null}},
    "issue_types": ["empty_shelf"],
    "severity_threshold": "low",
    "fill_rate_threshold": null,
    "location_filter": "any"
  }},
  "action": {{
    "alert_level": "info|warning|critical",
    "notification_message": "template da mensagem com {{zone_id}}, {{issue_type}}, {{severity}}"
  }},
  "validation": {{
    "is_valid": true,
    "ambiguities": ["lista de aspectos não claros"],
    "assumptions": ["lista de pressupostos assumidos"]
  }}
}}

Notas:
- zone_filter: lista de zonas específicas, ou [] para todas as zonas
- time_filter: horas de início e fim (0-23), ou null se sem filtro de hora
- issue_types: um ou mais de [empty_shelf, wrong_product, damaged, misaligned, label_missing, other]
- severity_threshold: nível mínimo que dispara a regra (low = qualquer, medium = médio+, high = só alto)
- fill_rate_threshold: número entre 0.0 e 1.0, ou null se não aplicável
- location_filter: "top" | "middle" | "bottom" | "any"
- alert_level: "info" (informativo), "warning" (aviso), "critical" (urgente)
- Se a regra for ambígua, lista as ambiguidades mas tenta uma conversão razoável (is_valid: true com assumptions)
- Só usa is_valid: false se a regra for completamente incompreensível
"""


def _call_gemini_text(prompt: str, model_name: str = "gemini-1.5-flash") -> str:
    import google.generativeai as genai
    import time

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY não definida.")
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(model_name)
    for attempt in range(4):
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            if "429" in str(e) and attempt < 3:
                time.sleep(2 ** (attempt + 1))
            else:
                raise


def _extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            return json.loads(match.group())
    raise ValueError("Não foi possível extrair JSON da resposta.")


def _load_all_rules() -> list[dict]:
    rules = []
    for f in sorted(RULES_DIR.glob("RULE_*.json")):
        with open(f) as fp:
            rules.append(json.load(fp))
    return rules


def _save_rule(rule: dict):
    path = RULES_DIR / f"{rule['rule_id']}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rule, f, ensure_ascii=False, indent=2)
    print(f"[OK] Regra guardada: {path.name}")


def add_rule(natural_language: str, interactive: bool = True) -> dict:
    """Converte texto em linguagem natural para uma regra estruturada."""
    existing = _load_all_rules()
    rule_id = f"RULE_{len(existing) + 1:03d}"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    prompt = (RULE_CONVERSION_PROMPT
              .replace("{RULE_TEXT}", natural_language)
              .replace("{RULE_ID}", rule_id)
              .replace("{TIMESTAMP}", timestamp))

    print(f"[Rule Engine] Convertendo regra para JSON...")
    raw = _call_gemini_text(prompt)
    rule = _extract_json(raw)

    rule.setdefault("rule_id", rule_id)
    rule.setdefault("created_at", timestamp)
    rule.setdefault("natural_language", natural_language)
    rule.setdefault("validation", {"is_valid": True, "ambiguities": [], "assumptions": []})

    ambiguities = rule.get("validation", {}).get("ambiguities", [])
    assumptions = rule.get("validation", {}).get("assumptions", [])

    print(f"\n{'='*50}")
    print(f"Regra convertida: {rule.get('description', '')}")
    if assumptions:
        print(f"\nPressupostos assumidos:")
        for a in assumptions:
            print(f"  - {a}")

    if ambiguities:
        print(f"\nAmbiguidades detectadas:")
        for i, amb in enumerate(ambiguities, 1):
            print(f"  {i}. {amb}")

        if interactive:
            print("\nDeseja guardar a regra com estes pressupostos? (s/n)")
            answer = input("> ").strip().lower()
            if answer != "s":
                print("[Cancelado] Regra não guardada. Por favor reformula a regra.")
                return rule

    _save_rule(rule)
    return rule


def list_rules() -> list[dict]:
    rules = _load_all_rules()
    if not rules:
        print("Sem regras definidas.")
        return []
    for r in rules:
        status = "✓" if r.get("validation", {}).get("is_valid", True) else "✗"
        print(f"[{status}] {r['rule_id']} — {r.get('description', r.get('natural_language', ''))}")
        conds = r.get("conditions", {})
        print(f"      Zonas: {conds.get('zone_filter') or 'todas'} | Alert: {r.get('action', {}).get('alert_level', '?')}")
    return rules


def delete_rule(rule_id: str) -> bool:
    path = RULES_DIR / f"{rule_id}.json"
    if path.exists():
        path.unlink()
        print(f"[OK] Regra {rule_id} removida.")
        return True
    print(f"[ERRO] Regra {rule_id} não encontrada.")
    return False


def _rule_matches(rule: dict, inspection: dict) -> tuple[bool, list[dict]]:
    conditions = rule.get("conditions", {})
    triggered_issues = []

    zone_filter = conditions.get("zone_filter", [])
    if zone_filter and inspection.get("zone_id") not in zone_filter:
        return False, []

    time_filter = conditions.get("time_filter", {})
    if time_filter and time_filter.get("hours_start") is not None:
        ts = inspection.get("timestamp", "")
        if ts:
            try:
                hour = int(ts[11:13])
                h_start = time_filter["hours_start"]
                h_end = time_filter.get("hours_end", 23)
                if not (h_start <= hour <= h_end):
                    return False, []
            except (ValueError, IndexError):
                pass

    fill_threshold = conditions.get("fill_rate_threshold")
    if fill_threshold is not None:
        if inspection.get("shelf_fill_rate", 1.0) >= fill_threshold:
            return False, []

    issue_types = conditions.get("issue_types", [])
    severity_map = {"low": 0, "medium": 1, "high": 2}
    sev_threshold = severity_map.get(conditions.get("severity_threshold", "low"), 0)
    location_filter = conditions.get("location_filter", "any")

    issues = inspection.get("issues", [])

    if not issues and not fill_threshold:
        return False, []

    for issue in issues:
        if issue_types and issue.get("type") not in issue_types:
            continue
        issue_sev = severity_map.get(issue.get("severity", "low"), 0)
        if issue_sev < sev_threshold:
            continue
        if location_filter and location_filter != "any":
            loc = issue.get("location", "").lower()
            if location_filter not in loc:
                continue
        triggered_issues.append(issue)

    if fill_threshold is not None and inspection.get("shelf_fill_rate", 1.0) < fill_threshold:
        if not triggered_issues:
            triggered_issues = [{"type": "fill_rate", "description": "Fill rate abaixo do limiar"}]

    return bool(triggered_issues), triggered_issues


def execute_rules(inspection: dict) -> list[dict]:
    """Executa as regras ativas sobre uma inspeção e gera notificações."""
    rules = _load_all_rules()
    notifications = []

    print(f"\n[Rule Engine] Verificando {len(rules)} regras para {inspection.get('inspection_id')}...")

    for rule in rules:
        if not rule.get("validation", {}).get("is_valid", True):
            continue

        fired, triggered_issues = _rule_matches(rule, inspection)

        if fired:
            action = rule.get("action", {})
            msg_template = action.get("notification_message", "Regra {rule_id} disparou.")

            notification_msg = (msg_template
                                .replace("{rule_id}", rule.get("rule_id", ""))
                                .replace("{zone_id}", inspection.get("zone_id", ""))
                                .replace("{inspection_id}", inspection.get("inspection_id", ""))
                                .replace("{fill_rate}", str(inspection.get("shelf_fill_rate", "")))
                                .replace("{issue_type}", triggered_issues[0].get("type", "") if triggered_issues else "")
                                .replace("{severity}", triggered_issues[0].get("severity", "") if triggered_issues else ""))

            notif = {
                "rule_id": rule["rule_id"],
                "rule_description": rule.get("description", ""),
                "alert_level": action.get("alert_level", "info"),
                "message": notification_msg,
                "inspection_id": inspection.get("inspection_id"),
                "zone_id": inspection.get("zone_id"),
                "triggered_by": triggered_issues,
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            notifications.append(notif)
            print(f"  [DISPARO] {rule['rule_id']} — Alert: {notif['alert_level'].upper()}")
            print(f"    {notification_msg}")
        else:
            print(f"  [OK] {rule['rule_id']} — não disparou")

    return notifications


def test_rule(rule_id: str, inspection: dict) -> dict:
    path = RULES_DIR / f"{rule_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Regra {rule_id} não encontrada.")
    with open(path) as f:
        rule = json.load(f)

    fired, triggered_issues = _rule_matches(rule, inspection)
    return {
        "rule_id": rule_id,
        "fired": fired,
        "triggered_issues": triggered_issues,
        "rule": rule,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso:")
        print("  python rule_engine.py add \"<regra em português>\"")
        print("  python rule_engine.py list")
        print("  python rule_engine.py delete <RULE_ID>")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "add" and len(sys.argv) > 2:
        add_rule(sys.argv[2])
    elif cmd == "list":
        list_rules()
    elif cmd == "delete" and len(sys.argv) > 2:
        delete_rule(sys.argv[2])
    else:
        print(f"Comando desconhecido: {cmd}")