# TP2 — Retail Vision Intelligence System

> **LIACD 2025/2026 | Universidade da Beira Interior**
> Análise visual de prateleiras de supermercado com modelos de linguagem multimodais

---

## Índice

1. [Descrição do Sistema](#1-descrição-do-sistema)
2. [Estrutura do Projecto](#2-estrutura-do-projecto)
3. [Instalação](#3-instalação)
4. [Configuração](#4-configuração)
5. [Uso — Interface CLI](#5-uso--interface-cli)
6. [Componentes](#6-componentes)
7. [Dataset](#7-dataset)
8. [Avaliação](#8-avaliação)
9. [Integração com o Projecto 1](#9-integração-com-o-projecto-1)
10. [Notas Técnicas](#10-notas-técnicas)

---

## 1. Descrição do Sistema

O **Retail Vision Intelligence System** complementa os dados de trajectória do Projecto 1 com análise visual do estado físico das prateleiras. O sistema recebe imagens de prateleiras, analisa-as com o Google Gemini 2.5 Flash Lite, detecta problemas operacionais (prateleiras vazias, produtos fora de posição, violações de planograma), e permite ao gestor de loja definir regras de detecção em linguagem natural.

O histórico de inspecções é indexado numa base de dados vectorial (ChromaDB), permitindo recuperação semântica de padrões passados para contextualizar análises futuras.

**Pipeline principal:**

```
Imagens de Prateleiras
        ↓
[1] shelf_inspector.py   — análise visual (Gemini 2.5 Flash Lite)
        ↓
[2] rule_engine.py       — regras em linguagem natural → JSON + execução
        ↓
[3] rag_memory.py        — indexação ChromaDB + recuperação semântica
        ↓
[4] report_generator.py  — relatórios Markdown com contexto histórico
        ↓
[5] interface.py         — CLI conversacional para o gestor de loja
```

---

## 2. Estrutura do Projecto

```
tp2/
├── README.md
├── requirements.txt
├── .env.example              ← variáveis de ambiente (copia para .env)
├── .gitignore
├── evaluate.py               ← harness de avaliação (único comando)
├── data/
│   ├── images/               ← dataset de imagens (≥500 imagens)
│   ├── inspections/          ← inspection records gerados automaticamente
│   ├── rules/                ← regras persistidas (RULE_001.json, ...)
│   ├── ground_truth.json     ← anotações manuais para avaliação
│   └── p1/                   ← dados do Projecto 1
│       ├── journeys.csv
│       ├── metrics.json
│       └── insights.json
├── src/
│   ├── shelf_inspector.py
│   ├── rule_engine.py
│   ├── rag_memory.py
│   ├── report_generator.py
│   └── interface.py
├── prompts/
│   ├── inspector_zero_shot.txt
│   ├── inspector_cot.txt
│   └── inspector_few_shot.txt
├── vectorstore/              ← ChromaDB persistente (gerado em runtime)
└── cache/                    ← cache MD5 de resultados da API
```

---

## 3. Instalação

```bash
# 1. Entra na pasta do projecto
cd tp2

# 2. Instala as dependências Python
pip install -r requirements.txt
```

**Dependências principais:**

| Pacote | Versão | Função |
|---|---|---|
| `google-generativeai` | ≥0.7.0 | API Gemini 2.5 Flash Lite |
| `chromadb` | ≥0.5.0 | Vector store persistente |
| `sentence-transformers` | ≥3.0.0 | Embeddings multilinguais (MiniLM-L12-v2) |
| `python-dotenv` | ≥1.0.0 | Variáveis de ambiente |
| `rich` | ≥13.0.0 | Interface CLI colorida |
| `pillow` | ≥10.0.0 | Processamento de imagens |
| `pandas` | ≥2.0.0 | Dados do Projecto 1 |

---

## 4. Configuração

```bash
# Copia o template de variáveis de ambiente
cp .env.example .env
```

Edita o ficheiro `.env` e preenche:

```env
# Obter em https://aistudio.google.com → Get API Key (gratuita)
GEMINI_API_KEY=a_tua_chave_aqui

# Caminhos (defaults já configurados)
CACHE_DIR=./cache
VECTORSTORE_DIR=./vectorstore
RULES_DIR=./data/rules
INSPECTIONS_DIR=./data/inspections

# Integração com o Projecto 1 (opcional)
P1_JOURNEYS_CSV=./data/p1/journeys.csv
P1_METRICS_JSON=./data/p1/metrics.json
P1_INSIGHTS_JSON=./data/p1/insights.json
```

> ⚠️ **Nunca commitas o ficheiro `.env`** — o `.gitignore` já o exclui. A chave de API é pessoal.

---

## 5. Uso — Interface CLI

```bash
cd src
python interface.py
```

### 5.1 Inspecção de Imagens

```
# Imagem única
> inspect Z_S3 --image ../data/images/shelf1.jpg
> inspect Z_S4 --image ../data/images/shelf2.jpg --strategy zero_shot
> inspect Z_S4 --image ../data/images/shelf2.jpg --strategy cot
> inspect Z_S4 --image ../data/images/shelf2.jpg --strategy few_shot

# Batch (todas as imagens numa pasta)
> inspect all --images-dir ../data/images/
```

**Estratégias disponíveis:**

| Estratégia | Flag | Descrição |
|---|---|---|
| Zero-Shot | `zero_shot` | Instrução directa sem exemplos — maior parse rate |
| Chain-of-Thought | `cot` (default) | Raciocínio por passos — melhor qualidade |
| Few-Shot | `few_shot` | Com exemplos textuais — descrições calibradas |

### 5.2 Gestão de Regras

```
# Adiciona uma regra em português natural
> add rule "Avisa-me quando a prateleira inferior estiver mais de 40% vazia"
> add rule "Na zona Z_S4, qualquer produto danificado é crítico"
> add rule "Quando o fill rate cair abaixo de 60% entre as 10h e as 13h, avisa-me"

# Lista regras activas
> list rules

# Remove uma regra
> delete rule RULE_001

# Testa uma regra numa imagem sem guardar notificação
> test rule RULE_001 --image ../data/images/shelf1.jpg
```

### 5.3 Consulta Histórica (RAG)

```
> history "quais as zonas com mais problemas esta semana?"
> history "última vez que Z_S1 teve fill rate abaixo de 50%?"
> history "existe algum padrão nos problemas às sextas-feiras?" --zone Z_S3
> compare Z_S1 Z_S4

# Indexa manualmente inspeções guardadas em disco
> rag index
> rag index --dir ../data/inspections/
```

### 5.4 Relatórios

```
# Relatório da sessão actual
> report --session today
> report --session today --output ../relatorio_hoje.md

# Relatório focado numa zona
> report --zone Z_S4
> report --zone Z_S4 --period "last 14 days"
```

### 5.5 Outros Comandos

```
> help    # mostra todos os comandos
> exit    # sai da interface
```

---

## 6. Componentes

### 6.1 Shelf Inspector (`src/shelf_inspector.py`)

Analisa imagens de prateleiras com o Gemini 2.5 Flash Lite. Funcionalidades:

- **Cache MD5**: imagens já analisadas não consomem quota — o resultado é carregado do disco
- **Rate limiting**: delay automático de 4,5s entre chamadas (~13 req/min < limite de 15)
- **Backoff exponencial**: em erros 429, aguarda 2s, 4s, 8s, 16s antes de desistir
- **Fallback gracioso**: quota esgotada notifica claramente, continua para imagens em cache

**Execução directa:**
```bash
cd src
python shelf_inspector.py ../data/images/shelf1.jpg Z_S4 cot
python shelf_inspector.py ../data/images/shelf1.jpg Z_S4 zero_shot
python shelf_inspector.py ../data/images/shelf1.jpg Z_S4 few_shot
```

**Schema de output:**
```json
{
  "inspection_id": "INS_20260613_143022_001",
  "timestamp": "2026-06-13T14:30:22Z",
  "image_path": "data/images/shelf1.jpg",
  "zone_id": "Z_S4",
  "overall_status": "warning",
  "issues": [
    {
      "issue_id": "ISS_001",
      "type": "empty_shelf",
      "location": "prateleira inferior, lado esquerdo",
      "severity": "medium",
      "description": "Espaço vazio de ~30% na prateleira inferior",
      "confidence": 0.87,
      "affected_area_pct": 0.12
    }
  ],
  "shelf_fill_rate": 0.75,
  "products_detected": ["gel de banho", "sabão"],
  "model_reasoning": "..."
}
```

### 6.2 Rule Engine (`src/rule_engine.py`)

Converte regras em linguagem natural para JSON estruturado e executa-as sobre os resultados das inspecções.

**Execução directa:**
```bash
cd src
python rule_engine.py add "Avisa quando fill rate abaixo de 60%"
python rule_engine.py list
python rule_engine.py delete RULE_001
```

### 6.3 RAG Memory (`src/rag_memory.py`)

Sistema de memória com ChromaDB e embeddings `paraphrase-multilingual-MiniLM-L12-v2`.

- **Chunking híbrido**: summary por inspecção + metadados estruturados para filtragem pré-retrieval
- **Integração P1**: summaries enriquecidos com contexto de afluência do Projecto 1
- **Similaridade de cosseno** com top-k configurável (k=3 por defeito)

**Execução directa:**
```bash
cd src
python rag_memory.py index                    # indexa data/inspections/
python rag_memory.py query "prateleira vazia zona Z_S4"
python rag_memory.py count
```

### 6.4 Report Generator (`src/report_generator.py`)

Gera relatórios Markdown com 6 secções: sumário executivo, problemas por zona, regras disparadas, contexto histórico RAG, recomendações e integração com Projecto 1.

**Execução directa:**
```bash
cd src
python report_generator.py session ../data/inspections/INS_001.json output.md
python report_generator.py zone Z_S4
```

### 6.5 Interface (`src/interface.py`)

CLI com `shlex.split` para parsing correcto de argumentos com espaços e aspas. Mantém estado de sessão entre comandos.

---

## 7. Dataset

O dataset foi construído a partir de fontes públicas:

| Fonte | Imagens | Licença |
|---|---|---|
| Roboflow — Supermarket Empty Shelf Detector | ~497 | CC BY 4.0 |
| HuggingFace — UniDataPro/grocery-shelves | ~400 | Académica |
| Roboflow — fyp-nrna1/empty-shelf-detector | ~80 | CC BY 4.0 |
| **Total** | **~975** | |

**Distribuição por categoria (aproximada):**

| Categoria | Imagens | Fonte principal |
|---|---|---|
| Prateleira vazia | ~500 | Roboflow (Out-of-Stock) |
| Prateleira normal | ~400 | HuggingFace |
| Casos mistos | ~75 | Roboflow fyp-nrna1 |

> **Nota:** As categorias violação de planograma, prateleira suja e caso ambíguo não foram suficientemente cobertas por fontes públicas. Esta limitação afecta as métricas nessas classes.

---

## 8. Avaliação

```bash
# Execução completa com ground truth
python evaluate.py --images-dir data/images/ --gt data/ground_truth.json --output evaluation_report.json

# Só uma estratégia
python evaluate.py --images-dir data/images/ --gt data/ground_truth.json --strategies cot --output eval_cot.json

# Sem ground truth (métricas parciais)
python evaluate.py --images-dir data/images/ --output evaluation_report.json
```

**Métricas calculadas:**

| Categoria | Métrica | Descrição |
|---|---|---|
| Visual | Issue Detection Rate | % de issues do GT correctamente identificados |
| Visual | False Positive Rate | % de issues reportados que não existem no GT |
| Visual | Severity Accuracy | % com severidade correctamente classificada |
| Visual | JSON Parse Rate | % de respostas JSON válidas |
| Visual | Hallucination Rate | % de afirmações não verificáveis na imagem |
| RAG | Recall@3 | % de queries com doc relevante no top-3 |
| RAG | Faithfulness | % de afirmações RAG suportadas pelos chunks |
| Rule Engine | Rule Parse Rate | % de regras convertidas para JSON válido |
| Rule Engine | Ambiguity Detection | % de regras ambíguas correctamente sinalizadas |
| Qualitativo | LLM-as-Judge | Score 0–1 com justificação automática |

**Formato do `ground_truth.json`:**
```json
[
  {
    "image": "nome_do_ficheiro.jpg",
    "zone_id": "Z_S4",
    "overall_status": "warning",
    "issues": [
      { "type": "empty_shelf", "severity": "medium", "location": "prateleira inferior" }
    ],
    "fill_rate": 0.65
  }
]
```

---

## 9. Integração com o Projecto 1

Coloca os ficheiros do Projecto 1 em `data/p1/` e configura o `.env`:

```env
P1_JOURNEYS_CSV=./data/p1/journeys.csv
P1_METRICS_JSON=./data/p1/metrics.json
P1_INSIGHTS_JSON=./data/p1/insights.json
```

O sistema integra automaticamente:

- **Summaries RAG**: enriquecidos com total de visitas, dwell médio e anomalias de tráfego da zona
- **Relatórios**: secção 6 correlaciona issues visuais com padrões de afluência
- **Contexto operacional**: distingue ruptura de stock por procura elevada vs. falha de reposição

---

## 10. Notas Técnicas

### Modelo

O sistema usa o **Google Gemini 2.5 Flash Lite** (substituição necessária — o `gemini-1.5-flash` foi descontinuado pela Google com erro 404 na API v1beta). Limites do free tier:

- 15 req/min (RPM)
- 1.000 req/dia

O rate limiting automático (4,5s entre chamadas) respeita estes limites. Com 3 estratégias × 18 imagens = 54 chamadas por ciclo completo de avaliação.

### Cache

Todos os resultados são guardados em `cache/` por hash MD5 da imagem. Imagens já analisadas não consomem quota, independentemente do número de execuções.

### Reprodutibilidade

O Gemini 2.5 Flash Lite não expõe parâmetro de seed. Para reprodutibilidade mínima, o sistema usa `temperature=0` nos testes. Outputs podem variar ligeiramente entre execuções.

### Segurança

```bash
# Verifica que o .env não está a ser rastreado pelo git
git check-ignore -v .env
# Deve mostrar: .gitignore:1:.env   .env
```

---

*Retail Vision Intelligence System — TP2 LIACD 2025/2026 | Felipe Candido Nº54698*
