"""
evaluate.py — Harness de avaliação do TP2
Executa métricas sobre um conjunto de imagens de teste com ground truth.

Uso:
  python evaluate.py --images-dir test_images/ --output evaluation_report.json
  python evaluate.py --images-dir test_images/ --gt ground_truth.json --output report.json
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
load_dotenv()


# ── Métricas de análise visual ─────────────────────────────────────────────

def compute_visual_metrics(predictions: list[dict], ground_truth: list[dict]) -> dict:
    """
    Calcula métricas de análise visual.

    Ground truth format:
    [{"image": "shelf1.jpg", "issues": [{"type": "empty_shelf", "severity": "high", ...}], "fill_rate": 0.6}]
    """
    total = len(ground_truth)
    if total == 0:
        return {"error": "Ground truth vazio"}

    # Issue Detection Rate (recall)
    tp_issues = 0
    total_gt_issues = 0
    total_pred_issues = 0
    false_positives = 0
    severity_correct = 0
    severity_total = 0
    json_parse_successes = 0
    hallucinations = 0
    hallucination_total = 0

    for gt in ground_truth:
        img_name = Path(gt["image"]).name
        # Encontra a predição correspondente
        pred = next((p for p in predictions if Path(p.get("image_path", "")).name == img_name), None)

        if pred is None:
            continue

        # JSON Parse Rate
        if "error" not in pred:
            json_parse_successes += 1

        gt_issues = gt.get("issues", [])
        pred_issues = pred.get("issues", [])
        total_gt_issues += len(gt_issues)
        total_pred_issues += len(pred_issues)

        # Para cada issue ground truth, verifica se foi detectado
        for gt_iss in gt_issues:
            gt_type = gt_iss.get("type")
            # Considera detectado se o tipo bate com alguma predição
            matching_pred = next(
                (p for p in pred_issues if p.get("type") == gt_type), None
            )
            if matching_pred:
                tp_issues += 1
                # Severity accuracy
                if matching_pred.get("severity") == gt_iss.get("severity"):
                    severity_correct += 1
                severity_total += 1

        # False positive rate: issues preditos que não existem no GT
        for pred_iss in pred_issues:
            pred_type = pred_iss.get("type")
            if not any(g.get("type") == pred_type for g in gt_issues):
                false_positives += 1

        # Hallucination Rate (heurística: confiança muito alta para issue inexistente)
        reasoning = pred.get("model_reasoning", "")
        for pred_iss in pred_issues:
            hallucination_total += 1
            pred_type = pred_iss.get("type")
            if not any(g.get("type") == pred_type for g in gt_issues):
                # Predição sem suporte no GT
                if pred_iss.get("confidence", 0) > 0.8:
                    hallucinations += 1

    return {
        "issue_detection_rate": tp_issues / max(total_gt_issues, 1),
        "false_positive_rate": false_positives / max(total_pred_issues, 1),
        "severity_accuracy": severity_correct / max(severity_total, 1),
        "json_parse_rate": json_parse_successes / total,
        "hallucination_rate": hallucinations / max(hallucination_total, 1),
        "totals": {
            "images": total,
            "gt_issues": total_gt_issues,
            "pred_issues": total_pred_issues,
            "true_positives": tp_issues,
            "false_positives": false_positives,
        }
    }


def compute_rag_metrics(queries_with_gt: list[dict], k: int = 3) -> dict:
    """
    Calcula Recall@k para o sistema RAG.
    """
    from rag_memory import evaluate_recall_at_k
    return evaluate_recall_at_k(queries_with_gt, k=k)


def compute_rule_metrics(rules_test_cases: list[dict]) -> dict:
    """
    Avalia o Rule Engine sobre casos de teste.

    Format: [{"rule_text": "...", "is_ambiguous": bool, "expected_conditions": {...}}]
    """
    from rule_engine import _call_gemini_text, _extract_json, RULE_CONVERSION_PROMPT
    import re

    parse_successes = 0
    correctness_successes = 0
    ambiguity_detected = 0
    ambiguous_count = 0

    for i, tc in enumerate(rules_test_cases):
        rule_text = tc["rule_text"]
        is_ambiguous = tc.get("is_ambiguous", False)

        if is_ambiguous:
            ambiguous_count += 1

        prompt = (RULE_CONVERSION_PROMPT
                  .replace("{RULE_TEXT}", rule_text)
                  .replace("{RULE_ID}", f"RULE_EVAL_{i:03d}")
                  .replace("{TIMESTAMP}", datetime.now().isoformat()))

        try:
            raw = _call_gemini_text(prompt)
            rule_json = _extract_json(raw)
            parse_successes += 1

            # Verifica ambiguidade
            ambigs = rule_json.get("validation", {}).get("ambiguities", [])
            if is_ambiguous and ambigs:
                ambiguity_detected += 1

            # Verifica correctness (comparação de condições esperadas)
            expected = tc.get("expected_conditions", {})
            actual = rule_json.get("conditions", {})
            if expected:
                correct = all(
                    actual.get(k) == v
                    for k, v in expected.items()
                    if v is not None
                )
                if correct:
                    correctness_successes += 1
            else:
                correctness_successes += 1  # Sem expected = conta como correcto

        except Exception as e:
            print(f"[EVAL] Erro no caso {i}: {e}")

        time.sleep(4)  # Rate limiting

    return {
        "rule_parse_rate": parse_successes / max(len(rules_test_cases), 1),
        "rule_correctness": correctness_successes / max(len(rules_test_cases), 1),
        "ambiguity_detection": ambiguity_detected / max(ambiguous_count, 1),
        "totals": {
            "total_rules": len(rules_test_cases),
            "parsed": parse_successes,
            "correct": correctness_successes,
            "ambiguous_total": ambiguous_count,
            "ambiguous_detected": ambiguity_detected,
        }
    }


def llm_as_judge(report_md: str, criteria: str) -> dict:
    """
    Usa o Gemini Flash como juiz para avaliação qualitativa.

    Returns:
        {"score": 0.0-1.0, "justification": str, "criteria": str}
    """
    from shelf_inspector import _call_gemini_with_backoff
    import google.generativeai as genai

    api_key = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=api_key)

    judge_prompt = f"""Você é um avaliador especializado de sistemas de inteligência de retalho.
Avalia o seguinte output do sistema com base nos critérios indicados.

CRITÉRIOS DE AVALIAÇÃO:
{criteria}

OUTPUT DO SISTEMA:
{report_md[:3000]}

Responde APENAS com JSON válido:
{{
  "score": 0.0,
  "justification": "explicação em 2-3 frases",
  "strengths": ["ponto forte 1", "ponto forte 2"],
  "weaknesses": ["ponto fraco 1", "ponto fraco 2"],
  "criteria": "{criteria[:50]}"
}}

Score de 0.0 (muito fraco) a 1.0 (excelente). Sê honesto e rigoroso."""

    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    try:
        response = gemini_model.generate_content(judge_prompt)
        result = json.loads(response.text.strip())
        return result
    except Exception as e:
        return {"score": 0.0, "justification": f"Erro: {e}", "criteria": criteria}


def run_full_evaluation(
    images_dir: str,
    ground_truth_path: Optional[str] = None,
    output_path: str = "evaluation_report.json",
    strategies: list[str] = None,
    k_rag: int = 3,
) -> dict:
    """
    Executa o harness completo de avaliação.

    Args:
        images_dir: Directoria com imagens de teste
        ground_truth_path: JSON com ground truth (opcional, gera avaliação parcial sem)
        output_path: Ficheiro de output
        strategies: Lista de estratégias a avaliar (default: todas)
        k_rag: k para Recall@k no RAG
    """
    from shelf_inspector import inspect_image
    from rag_memory import index_inspection, query

    if strategies is None:
        strategies = ["zero_shot", "cot", "few_shot"]

    print(f"\n{'='*60}")
    print("HARNESS DE AVALIAÇÃO — Retail Vision Intelligence TP2")
    print(f"{'='*60}")
    print(f"Directoria: {images_dir}")
    print(f"Estratégias: {strategies}")
    print(f"Output: {output_path}\n")

    images = list(Path(images_dir).glob("*.jpg")) + \
             list(Path(images_dir).glob("*.jpeg")) + \
             list(Path(images_dir).glob("*.png"))

    if not images:
        print("[ERRO] Sem imagens encontradas.")
        return {}

    print(f"[Avaliação] {len(images)} imagens encontradas.")

    # Carrega ground truth se disponível
    ground_truth = []
    if ground_truth_path and Path(ground_truth_path).exists():
        with open(ground_truth_path) as f:
            ground_truth = json.load(f)
        print(f"[GT] {len(ground_truth)} ground truth carregados.")

    # ── Fase 1: Inspecção com cada estratégia ─────────────────────────────
    results_by_strategy = {}

    for strategy in strategies:
        print(f"\n[Estratégia: {strategy.upper()}]")
        strategy_results = []

        for i, img in enumerate(images[:10]):  # Máx 10 imagens por estratégia
            zone = "Z_S1"  # Zona default para teste
            # Infere zona do nome do ficheiro se possível (e.g. "Z_S3_shelf1.jpg")
            name = img.stem
            import re
            m = re.search(r'Z_[A-Z]\d+', name)
            if m:
                zone = m.group()

            print(f"  [{i+1}/{min(len(images), 10)}] {img.name} (zona {zone})")
            try:
                result = inspect_image(str(img), zone_id=zone, strategy=strategy)
                strategy_results.append(result)
                # Indexa no RAG para avaliação do RAG
                if strategy == "cot":
                    index_inspection(result)
            except Exception as e:
                print(f"  [ERRO] {e}")
                strategy_results.append({"image_path": str(img), "error": str(e)})

            if i < min(len(images), 10) - 1:
                time.sleep(4)

        results_by_strategy[strategy] = strategy_results

    # ── Fase 2: Métricas visuais ───────────────────────────────────────────
    print("\n[Métricas] Calculando métricas visuais...")
    visual_metrics = {}

    for strategy, preds in results_by_strategy.items():
        if ground_truth:
            metrics = compute_visual_metrics(preds, ground_truth)
        else:
            # Sem GT: calcula apenas JSON parse rate e estatísticas
            parse_rate = sum(1 for p in preds if "error" not in p) / max(len(preds), 1)
            avg_issues = sum(len(p.get("issues", [])) for p in preds) / max(len(preds), 1)
            metrics = {
                "json_parse_rate": parse_rate,
                "avg_issues_per_image": avg_issues,
                "note": "Ground truth não fornecido — métricas completas indisponíveis"
            }
        visual_metrics[strategy] = metrics
        print(f"  {strategy}: JSON parse rate = {metrics.get('json_parse_rate', 0):.0%}")

    # ── Fase 3: Métricas RAG ───────────────────────────────────────────────
    print("\n[RAG] Avaliando Recall@k...")
    # Queries de teste predefinidas (o aluno deve definir ground truth para as suas inspeções)
    rag_test_queries = [
        {
            "query": "prateleira vazia zona produto",
            "relevant_ids": []  # Preencher com IDs reais após indexação
        },
        {
            "query": "produto tombado misaligned",
            "relevant_ids": []
        },
    ]
    # Filtra queries sem ground truth real
    rag_queries_with_gt = [q for q in rag_test_queries if q["relevant_ids"]]
    rag_metrics = {}
    if rag_queries_with_gt:
        rag_metrics = compute_rag_metrics(rag_queries_with_gt, k=k_rag)
        print(f"  Recall@{k_rag} = {rag_metrics.get(f'Recall@{k_rag}', 0):.2f}")
    else:
        rag_metrics = {"note": "Sem ground truth RAG definido para estas imagens"}
        print("  [AVISO] Sem ground truth RAG. Define relevant_ids em rag_test_queries.")

    # ── Fase 4: LLM-as-Judge ──────────────────────────────────────────────
    print("\n[LLM-as-Judge] Avaliando qualidade dos outputs...")
    judge_results = {}

    # Avalia um relatório de exemplo com a estratégia CoT
    cot_results = results_by_strategy.get("cot", [])
    if cot_results:
        from report_generator import generate_inspection_report
        sample_report = generate_inspection_report(
            cot_results[:3],
            session_name="avaliação automática"
        )

        judge_criteria = [
            "Clareza e accionabilidade das recomendações (são específicas e executáveis sem interpretação adicional?)",
            "Precisão das descrições de issues (as descrições são concretas e baseadas em observações visuais?)",
            "Qualidade do sumário executivo (é conciso, directo e útil para um gestor de loja?)",
        ]

        for criteria in judge_criteria:
            print(f"  Critério: {criteria[:50]}...")
            result = llm_as_judge(sample_report, criteria)
            judge_results[criteria[:50]] = result
            print(f"  Score: {result.get('score', 0):.2f}")
            time.sleep(4)

    # ── Fase 5: Comparação de estratégias ─────────────────────────────────
    strategy_comparison = {}
    for s, metrics in visual_metrics.items():
        strategy_comparison[s] = {
            "json_parse_rate": metrics.get("json_parse_rate", 0),
            "issue_detection_rate": metrics.get("issue_detection_rate", 0),
            "false_positive_rate": metrics.get("false_positive_rate", 0),
            "severity_accuracy": metrics.get("severity_accuracy", 0),
        }

    # ── Relatório final ────────────────────────────────────────────────────
    evaluation_report = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "images_evaluated": len(images[:10]),
            "strategies_evaluated": strategies,
            "k_rag": k_rag,
            "has_ground_truth": bool(ground_truth),
        },
        "visual_metrics_by_strategy": visual_metrics,
        "strategy_comparison": strategy_comparison,
        "rag_metrics": rag_metrics,
        "llm_judge_results": judge_results,
        "raw_results": {
            s: [
                {
                    "image": r.get("image_path"),
                    "status": r.get("overall_status"),
                    "fill_rate": r.get("shelf_fill_rate"),
                    "n_issues": len(r.get("issues", [])),
                    "json_valid": "error" not in r,
                }
                for r in results
            ]
            for s, results in results_by_strategy.items()
        }
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_report, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"AVALIAÇÃO CONCLUÍDA")
    print(f"Relatório guardado em: {output_path}")
    print(f"{'='*60}")

    # Resumo rápido
    for s, m in visual_metrics.items():
        print(f"  [{s}] JSON parse: {m.get('json_parse_rate', 0):.0%} | "
              f"Detection: {m.get('issue_detection_rate', 'N/A')} | "
              f"FP rate: {m.get('false_positive_rate', 'N/A')}")

    return evaluation_report


# ── Tipagem ────────────────────────────────────────────────────────────────
from typing import Optional


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Harness de avaliação — Retail Vision Intelligence TP2"
    )
    parser.add_argument("--images-dir", required=True, help="Directoria com imagens de teste")
    parser.add_argument("--gt", "--ground-truth", default=None,
                        help="JSON com ground truth (opcional)")
    parser.add_argument("--output", default="evaluation_report.json",
                        help="Ficheiro de output (default: evaluation_report.json)")
    parser.add_argument("--strategies", nargs="+",
                        default=["zero_shot", "cot", "few_shot"],
                        help="Estratégias a avaliar")
    parser.add_argument("--k", type=int, default=3,
                        help="k para Recall@k no RAG (default: 3)")

    args = parser.parse_args()

    run_full_evaluation(
        images_dir=args.images_dir,
        ground_truth_path=args.gt,
        output_path=args.output,
        strategies=args.strategies,
        k_rag=args.k,
    )
