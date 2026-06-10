# TP2 — Retail Vision Intelligence System

> LIACD 2025/2026 | Análise visual de prateleiras com LLMs multimodais

---

## Estrutura do Projecto

```
tp2/
├── README.md
├── requirements.txt
├── .env.example            ← copia para .env e preenche GEMINI_API_KEY
├── evaluate.py             ← harness de avaliação (único comando)
├── data/
│   ├── images/             ← dataset de imagens (≥500 imagens)
│   ├── inspections/        ← inspection records gerados automaticamente
│   ├── rules/              ← regras persistidas (RULE_001.json, ...)
│   └── p1/                 ← dados do Projecto 1 (opcional)
│       ├── journeys.csv
│       ├── metrics.json
│       └── insights.json
├── src/
│   ├── shelf_inspector.py  ← Componente 1: análise visual
│   ├── rule_engine.py      ← Componente 2: regras em linguagem natural
│   ├── rag_memory.py       ← Componente 3: memória RAG (ChromaDB)
│   ├── report_generator.py ← Componente 4: relatórios Markdown
│   └── interface.py        ← Componente 5: interface CLI
├── prompts/
│   ├── inspector_zero_shot.txt
│   ├── inspector_cot.txt
│   └── inspector_few_shot.txt
├── vectorstore/            ← ChromaDB persistente (gerado em runtime)
└── cache/                  ← cache de resultados API (hash MD5)
```

---

## Instalação

```bash
# 1. Clona e entra na directoria
cd tp2

# 2. Instala dependências
pip install -r requirements.txt

# 3. Configura variáveis de ambiente
cp .env.example .env
# Edita .env e preenche GEMINI_API_KEY (obtém em https://aistudio.google.com)

# 4. (Opcional) Integra dados do Projecto 1
cp /caminho/para/journeys.csv data/p1/journeys.csv
cp /caminho/para/metrics.json data/p1/metrics.json
cp /caminho/para/insights.json data/p1/insights.json
# Actualiza os caminhos em .env
```

---

## Uso

### Interface conversacional (modo principal)
```bash
cd src && python interface.py
```

### Comandos disponíveis

```
# Inspecção de imagem única
inspect Z_S3 --image shelf_photo.jpg
inspect Z_S3 --image shelf_photo.jpg --strategy zero_shot

# Inspecção em batch
inspect all --images-dir ./data/images/

# Definição de regras
add rule "Avisa-me quando a prateleira inferior estiver mais de 40% vazia"
add rule "Na zona Z_S1, se não houver produtos de laticínios visíveis, é crítico"
list rules
delete rule RULE_001
test rule RULE_001 --image shelf_photo.jpg

# Consulta histórica (RAG)
history "quais as zonas com mais problemas esta semana?"
history "ultima vez que Z_S1 teve fill rate abaixo de 50%?"
history "existe algum padrão nos problemas às sextas-feiras" --zone Z_S3
compare Z_S1 Z_S3
rag index                          # indexa todas as inspeções em disco

# Relatórios
report --session today
report --session today --output relatorio.md
report --zone Z_S3 --period "last 14 days"

# Ajuda
help
exit
```

### Inspecção directa (sem interface)
```bash
cd src
python shelf_inspector.py imagem.jpg Z_S3 cot
python shelf_inspector.py imagem.jpg Z_S3 zero_shot
python shelf_inspector.py imagem.jpg Z_S3 few_shot
```

### Harness de avaliação
```bash
# Com ground truth
python evaluate.py --images-dir test_images/ --gt ground_truth.json --output eval.json

# Sem ground truth (avaliação parcial)
python evaluate.py --images-dir test_images/ --output eval.json

# Só uma estratégia
python evaluate.py --images-dir test_images/ --strategies cot --output eval_cot.json
```

### Rule Engine
```bash
cd src
python rule_engine.py add "Avisa quando fill rate abaixo de 60%"
python rule_engine.py list
python rule_engine.py delete RULE_001
```

### RAG Memory
```bash
cd src
python rag_memory.py index                              # indexa data/inspections/
python rag_memory.py index --dir /outro/caminho
python rag_memory.py query "problemas em Z_S1 esta semana"
python rag_memory.py count
```

---

## Dataset de Imagens

O dataset deve conter ≥500 imagens nas seguintes categorias:

| Tipo | Mínimo | Fontes sugeridas |
|------|--------|-----------------|
| Prateleira normal | 150 | SKU-110K, Grocery Store Dataset |
| Prateleira vazia | 100 | GroZi-120, Open Images |
| Violação de planograma | 100 | SKU-110K |
| Suja / desordenada | 80 | Open Images |
| Ambígua | 70 | Recolha própria |

**Fontes:**
- SKU-110K: https://github.com/eg4000/SKU110K_CVPR19
- Grocery Store Dataset: HuggingFace `johnanvik/grocery-store-dataset`
- Open Images: https://storage.googleapis.com/openimages/web/index.html

---

## Ground Truth para Avaliação

Cria um ficheiro `ground_truth.json` para avaliação:

```json
[
  {
    "image": "test1.jpg",
    "zone_id": "Z_S3",
    "overall_status": "warning",
    "issues": [
      {
        "type": "empty_shelf",
        "location": "prateleira inferior",
        "severity": "medium"
      }
    ],
    "fill_rate": 0.65
  }
]
```

---

## Integração com Projecto 1

Copia os outputs do Projecto 1 para `data/p1/` e actualiza `.env`:

```env
P1_JOURNEYS_CSV=./data/p1/journeys.csv
P1_METRICS_JSON=./data/p1/metrics.json
P1_INSIGHTS_JSON=./data/p1/insights.json
```

O sistema irá automaticamente:
- Enriquecer summaries RAG com contexto de afluência
- Correlacionar issues visuais com anomalias de tráfego nos relatórios
- Identificar se prateleiras vazias coincidem com picos de afluência

---

## Limites de API

- **Gemini Flash gratuito:** 15 req/min, 1500 req/dia
- **Cache:** todos os resultados são guardados em `cache/` por hash MD5 — imagens já analisadas não consomem quota
- **Rate limiting:** delay automático de 4.5s entre chamadas (~13 req/min)
- **Backoff exponencial:** em erros 429, o sistema espera 2s, 4s, 8s, 16s antes de desistir

---

## Notas Técnicas

- Temperature=0.0 para reproducibilidade máxima (Gemini Flash não expõe seed)
- Embeddings: `paraphrase-multilingual-MiniLM-L12-v2` (local, suporta português)
- Vector store: ChromaDB persistente em `vectorstore/`
- Chunking: híbrido — summary por inspeção + metadados estruturados para filtragem
- Estratégia CoT é a default (melhor trade-off qualidade/quota)
