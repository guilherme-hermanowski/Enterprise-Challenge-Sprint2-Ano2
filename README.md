# FIAP - Inteligência artificial e data science

<p align="center">
<a href= "https://www.fiap.com.br/"><img src="assets/logo-fiap.png" alt="FIAP - Faculdade de Informática e Admnistração Paulista" border="0" width=40% height=40%></a>
</p>

<br>

# Enterprise Challenge - Sprint 1 - DASA
Cap 2 - Colheita de Dados e Insights - dados valiosos e maduros - Enterprise Challenge - Sprint 1

## Nome do grupo
41

## 👨‍🎓 Integrantes: 
- <a href="https://www.linkedin.com/company/inova-fusca">Guilherme Campos Hermanowski </a>
- <a href="https://www.linkedin.com/company/inova-fusca">Fatima Candal</a>
- <a href="https://www.linkedin.com/company/inova-fusca"> Matheus Alboredo Soares</a> 
- <a href="https://www.linkedin.com/company/inova-fusca">Jonathan Willian Luft </a>

## 👩‍🏫 Professores:
### Tutor(a) 
- <a href="https://www.linkedin.com/company/inova-fusca">Caique Nonato da Silva Bezerra</a>
### Coordenador(a)
- <a href="https://www.linkedin.com/company/inova-fusca">ANDRÉ GODOI CHIOVATO</a>

---

## Estrutura do Projeto

```
rag_genomics/
├── app.py                    # Interface Streamlit (Sprint 2 · UI)
├── rag_system.py             # Pipeline RAG (Sprint 2 · Inteligência)
├── sample_sprint1_output.json # JSON de exemplo da Sprint 1
├── requirements.txt          # Dependências
└── README.md                 # Este arquivo
```

---

## Instalação

```bash
# 1. Criar ambiente virtual
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate       # Windows

# 2. Instalar dependências base (interface mock)
pip install streamlit

# 3. Instalar dependências do RAG real (opcional)
pip install sentence-transformers faiss-cpu

# 4. Para uso com OpenAI ou Anthropic
pip install openai anthropic
```

---

## Executar a Interface

```bash
streamlit run app.py
```

Acesse: http://localhost:8501

---

## Integrar o RAG Real (Produção)

Substitua a função `mock_rag_answer()` em `app.py`:

```python
# No topo do app.py, importe o pipeline
from rag_system import GenomicRAGPipeline

# Inicialize uma vez (use @st.cache_resource)
@st.cache_resource
def load_pipeline():
    config = {
        "embedding_backend": "sentence_transformers",
        "embedding_model": "paraphrase-multilingual-mpnet-base-v2",
        "llm_backend": "openai",          # ou "anthropic"
        "llm_model": "gpt-4o-mini",       # ou "claude-sonnet-4-20250514"
        "top_k": 5,
        "min_score": 0.25,
    }
    return GenomicRAGPipeline.from_config(config)

# No handler de upload, indexe o JSON:
pipeline = load_pipeline()
pipeline.index_patient(genomic_json)

# Na função de resposta:
def real_rag_answer(question: str) -> dict:
    response = pipeline.answer(question)
    return {
        "answer": response.answer,
        "sources": [
            {
                "section": rc.chunk.section,
                "subsection": rc.chunk.subsection,
                "text": rc.chunk.text,
                "score": rc.score,
            }
            for rc in response.retrieved_chunks
        ],
    }
```

---

## Variáveis de Ambiente

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

---

## Arquitetura

```
Usuário → Pergunta
    ↓
[Streamlit UI] app.py
    ↓
[Retriever] Embedding da pergunta → Busca vetorial → Top-K chunks
    ↓
[Context Builder] Formata chunks como contexto estruturado
    ↓
[LLM] Prompt anti-alucinação + Contexto + Pergunta → Resposta
    ↓
[UI] Exibe resposta + Fontes rastreáveis + Disclaimer
```

---

## Decisões de Design

| Aspecto | Escolha | Razão |
|---------|---------|-------|
| Embedding | `paraphrase-multilingual-mpnet-base-v2` | Suporte nativo PT-BR, local, gratuito |
| Vector DB | FAISS (memória) | Zero infraestrutura, adequado para 1 paciente |
| LLM | GPT-4o-mini / Claude Sonnet | Custo-benefício, bom em PT-BR |
| Temperatura | 0.1 | Respostas factuais, minimiza alucinações |
| Chunking | Por domínio semântico | Cada doença/trait = 1 chunk = recuperação precisa |

---

## Anti-Alucinação

O sistema possui 3 camadas de proteção:

1. **Prompt de Sistema**: instrui explicitamente o LLM a responder *apenas* com base no contexto
2. **Score Mínimo**: chunks com similaridade < 0.25 são descartados
3. **Grounding Check**: heurística pós-geração verifica se a resposta usa termos do contexto
