"""
rag_memory.py — Componente 3 do TP2
Sistema de memória RAG com ChromaDB e sentence-transformers.
Indexa inspeções históricas e suporta queries em linguagem natural.
Inclui integração opcional com dados de trajectória do Projecto 1.
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

VECTORSTORE_DIR = Path(os.getenv("VECTORSTORE_DIR", "./vectorstore"))
INSPECTIONS_DIR = Path(os.getenv("INSPECTIONS_DIR", "./data/inspections"))
P1_JOURNEYS_CSV = os.getenv("P1_JOURNEYS_CSV", "")
P1_METRICS_JSON = os.getenv("P1_METRICS_JSON", "")

# Modelo de embeddings multilingual (suporta português)
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Coleção ChromaDB
COLLECTION_NAME = "shelf_inspections"


def _get_chroma_client():
    """Retorna cliente ChromaDB persistente."""
    import chromadb
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(VECTORSTORE_DIR))


def _get_embedding_model():
    """Carrega modelo de embeddings (cached após primeira carga)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL)


def _get_collection():
    """Retorna a coleção ChromaDB (cria se não existir)."""
    client = _get_chroma_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Inspeções de prateleiras de supermercado"}
    )


def _build_summary(inspection: dict, p1_context: Optional[str] = None) -> str:
    """
    Gera um summary rico em termos semanticamente relevantes para indexação.
    O campo summary é o texto que vai ter embeddings.

    Exemplo de bom summary:
    "prateleira inferior da zona Z_S3 com fill rate de 72%, produto de limpeza
    fora de posição na secção central, embalagem danificada detetada no lado direito,
    terça-feira 15h."
    """
    zone = inspection.get("zone_id", "zona desconhecida")
    status = inspection.get("overall_status", "ok")
    fill_rate = inspection.get("shelf_fill_rate", 1.0)
    products = ", ".join(inspection.get("products_detected", []))
    ts = inspection.get("timestamp", "")

    # Formata data e hora legível
    date_str = ""
    if ts:
        try:
            dt = datetime.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S")
            weekdays_pt = ["segunda-feira", "terça-feira", "quarta-feira",
                           "quinta-feira", "sexta-feira", "sábado", "domingo"]
            date_str = f"{weekdays_pt[dt.weekday()]} {dt.hour}h"
        except ValueError:
            date_str = ts[:10]

    # Descreve issues
    issues_text = []
    for iss in inspection.get("issues", []):
        issues_text.append(
            f"{iss.get('type', 'problema')} na {iss.get('location', 'prateleira')} "
            f"(severidade {iss.get('severity', 'low')}): {iss.get('description', '')}"
        )

    summary = (
        f"Zona {zone}, {date_str}. "
        f"Status: {status}. "
        f"Fill rate: {fill_rate:.0%}. "
        f"Produtos: {products}. "
    )
    if issues_text:
        summary += "Issues: " + "; ".join(issues_text) + ". "
    if p1_context:
        summary += f"Contexto de afluência (Projecto 1): {p1_context}. "

    return summary.strip()


def _get_p1_context(zone_id: str, timestamp: str) -> Optional[str]:
    """
    Obtém contexto de afluência do Projecto 1 para uma zona e hora específicas.
    Retorna None se os dados não estiverem disponíveis.
    """
    if not P1_METRICS_JSON or not Path(P1_METRICS_JSON).exists():
        return None
    if not P1_JOURNEYS_CSV or not Path(P1_JOURNEYS_CSV).exists():
        return None

    try:
        import pandas as pd

        with open(P1_METRICS_JSON) as f:
            metrics = json.load(f)

        # Verifica se a zona tem métricas
        zone_metrics = metrics.get("zone_metrics", {}).get(zone_id, {})
        if not zone_metrics:
            return None

        # Obtém hora da inspeção
        hour = None
        if timestamp:
            try:
                hour = int(timestamp[11:13])
            except (ValueError, IndexError):
                pass

        # Verifica anomalias na zona nessa hora
        anomalies = metrics.get("anomalies", [])
        zone_anomalies = [a for a in anomalies if a.get("zone") == zone_id]

        total_entries = zone_metrics.get("total_entries", 0)
        avg_dwell = zone_metrics.get("avg_dwell_s", 0)

        context_parts = [
            f"{zone_id} teve {total_entries} visitas na semana",
            f"dwell médio de {avg_dwell:.0f}s"
        ]

        if zone_anomalies:
            for a in zone_anomalies[:2]:
                direction = a.get("direction", "")
                actual = a.get("actual", 0)
                expected = a.get("expected", 0)
                context_parts.append(
                    f"anomalia às {a.get('hour')}h: {actual} visitantes "
                    f"({direction} do esperado de {expected:.1f})"
                )

        return "; ".join(context_parts)
    except Exception as e:
        print(f"[P1 Context] Erro ao carregar dados do Projecto 1: {e}")
        return None


def index_inspection(inspection: dict) -> str:
    """
    Indexa uma inspeção na vector store.

    Estratégia de chunking: híbrida
    - O summary da inspeção é o chunk principal (para queries semânticas)
    - Metadados estruturados permitem filtragem pré-retrieval

    Returns:
        ID do documento indexado
    """
    collection = _get_collection()
    model = _get_embedding_model()

    inspection_id = inspection.get("inspection_id", f"INS_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    zone_id = inspection.get("zone_id", "")
    timestamp = inspection.get("timestamp", "")

    # Contexto do Projecto 1
    p1_context = _get_p1_context(zone_id, timestamp)

    # Gera summary rico
    summary = _build_summary(inspection, p1_context)

    # Embedding
    embedding = model.encode(summary).tolist()

    # Metadados para filtragem pré-retrieval
    metadata = {
        "inspection_id": inspection_id,
        "zone_id": zone_id,
        "timestamp": timestamp[:10] if timestamp else "",
        "hour": int(timestamp[11:13]) if len(timestamp) >= 13 else -1,
        "overall_status": inspection.get("overall_status", "ok"),
        "shelf_fill_rate": float(inspection.get("shelf_fill_rate", 1.0)),
        "n_issues": len(inspection.get("issues", [])),
        "has_p1_context": p1_context is not None,
    }

    # Indexa na ChromaDB
    collection.upsert(
        ids=[inspection_id],
        embeddings=[embedding],
        documents=[summary],
        metadatas=[metadata]
    )

    print(f"[RAG] Indexada inspeção: {inspection_id} (zona {zone_id})")
    print(f"[RAG] Summary: {summary[:120]}...")
    return inspection_id


def index_all_inspections(inspections_dir: Optional[str] = None) -> int:
    """
    Indexa todas as inspeções guardadas em disco.

    Returns:
        Número de inspeções indexadas
    """
    dir_path = Path(inspections_dir or INSPECTIONS_DIR)
    files = list(dir_path.glob("INS_*.json"))
    count = 0

    print(f"[RAG] Indexando {len(files)} inspeções...")
    for f in sorted(files):
        try:
            with open(f) as fp:
                inspection = json.load(fp)
            index_inspection(inspection)
            count += 1
        except Exception as e:
            print(f"[AVISO] Erro ao indexar {f.name}: {e}")

    print(f"[RAG] {count}/{len(files)} inspeções indexadas.")
    return count


def query(
    natural_language_query: str,
    k: int = 3,
    zone_filter: Optional[str] = None,
) -> dict:
    """
    Query ao sistema de memória em linguagem natural.
    Usa o LLM para sintetizar uma resposta com base nos documentos recuperados.

    Args:
        natural_language_query: Pergunta em linguagem natural
        k: Número de documentos a recuperar
        zone_filter: Filtrar por zona específica (e.g. "Z_S1")

    Returns:
        Dicionário com resposta e documentos fonte
    """
    collection = _get_collection()
    model = _get_embedding_model()

    # Embedding da query
    query_embedding = model.encode(natural_language_query).tolist()

    # Prepara filtro (where clause ChromaDB)
    where = None
    if zone_filter:
        where = {"zone_id": {"$eq": zone_filter}}

    # Retrieval
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(k, collection.count()),
        where=where,
        include=["documents", "metadatas", "distances"]
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not docs:
        return {
            "query": natural_language_query,
            "answer": "Não foram encontradas inspeções relevantes no histórico.",
            "sources": []
        }

    # Contexto aumentado para o LLM
    context_parts = []
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances), 1):
        context_parts.append(
            f"[{i}] Inspeção {meta.get('inspection_id', '?')} "
            f"(zona {meta.get('zone_id', '?')}, {meta.get('timestamp', '?')}, "
            f"similaridade {1-dist:.2f}):\n{doc}"
        )
    context = "\n\n".join(context_parts)

    # Síntese com LLM
    synthesis_prompt = f"""Você é um assistente de gestão de loja de supermercado.
Responde à seguinte pergunta com base exclusivamente nos registos históricos de inspeção fornecidos.
Cita sempre os inspection_id relevantes na tua resposta.
Se os registos não forem suficientes para responder, diz claramente.

PERGUNTA: {natural_language_query}

REGISTOS RECUPERADOS:
{context}

Resposta concisa e directa em português (máx. 150 palavras):"""

    from shelf_inspector import _call_gemini_with_backoff
    import google.generativeai as genai

    api_key = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")

    try:
        response = gemini_model.generate_content(synthesis_prompt)
        answer = response.text.strip()
    except Exception as e:
        answer = f"[Erro ao sintetizar resposta: {e}]\n\nDocumentos recuperados:\n{context}"

    return {
        "query": natural_language_query,
        "answer": answer,
        "sources": [
            {
                "inspection_id": m.get("inspection_id"),
                "zone_id": m.get("zone_id"),
                "timestamp": m.get("timestamp"),
                "similarity": round(1 - d, 3),
                "summary": doc,
            }
            for m, d, doc in zip(metas, distances, docs)
        ]
    }


def evaluate_recall_at_k(
    queries_with_ground_truth: list[dict],
    k: int = 3,
) -> dict:
    """
    Avalia Recall@k sobre um conjunto de queries com ground truth.

    Args:
        queries_with_ground_truth: Lista de {"query": str, "relevant_ids": [str]}
        k: Número de documentos a recuperar

    Returns:
        Métricas de avaliação
    """
    collection = _get_collection()
    model = _get_embedding_model()

    hits = 0
    results_detail = []

    for item in queries_with_ground_truth:
        q = item["query"]
        relevant_ids = set(item["relevant_ids"])

        emb = model.encode(q).tolist()
        results = collection.query(
            query_embeddings=[emb],
            n_results=k,
            include=["metadatas"]
        )

        retrieved_ids = set(
            m.get("inspection_id", "")
            for m in results.get("metadatas", [[]])[0]
        )

        hit = bool(relevant_ids & retrieved_ids)
        hits += int(hit)
        results_detail.append({
            "query": q,
            "relevant_ids": list(relevant_ids),
            "retrieved_ids": list(retrieved_ids),
            "hit": hit
        })

    recall = hits / len(queries_with_ground_truth) if queries_with_ground_truth else 0.0

    return {
        f"Recall@{k}": recall,
        "hits": hits,
        "total_queries": len(queries_with_ground_truth),
        "detail": results_detail
    }


# --- CLI simples ---
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso:")
        print("  python rag_memory.py index [dir_inspeccoes]")
        print("  python rag_memory.py query \"<pergunta>\"")
        print("  python rag_memory.py count")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "index":
        d = sys.argv[2] if len(sys.argv) > 2 else None
        index_all_inspections(d)
    elif cmd == "query" and len(sys.argv) > 2:
        result = query(" ".join(sys.argv[2:]))
        print(f"\nPergunta: {result['query']}")
        print(f"\nResposta:\n{result['answer']}")
        print(f"\nFontes ({len(result['sources'])}):")
        for s in result["sources"]:
            print(f"  - {s['inspection_id']} (zona {s['zone_id']}, sim={s['similarity']})")
    elif cmd == "count":
        col = _get_collection()
        print(f"Total de inspeções indexadas: {col.count()}")
    else:
        print(f"Comando desconhecido: {cmd}")
