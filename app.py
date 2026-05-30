"""
=============================================================================
Genera Intelligence · Chat Interface — Challenge Dasa/Genera Sprint 2
=============================================================================
Interface de chat para consulta ao perfil genômico personalizado.
Integrada (mock-ready) com o pipeline RAG da Sprint 2.

Execução:
    streamlit run app.py

Dependências:
    pip install streamlit
    (Opcional para RAG real: sentence-transformers faiss-cpu openai anthropic)
=============================================================================
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import streamlit as st
import re

# ---------------------------------------------------------------------------
# Configuração da página (DEVE ser o primeiro comando Streamlit)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Genera Intelligence · Seu Perfil Genômico",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===========================================================================
# ESTILOS GLOBAIS — Design System Genera
# Paleta: Branco clínico + Verde-teal profundo + Âmbar como acento
# Tipografia: DM Serif Display (títulos) + DM Sans (corpo)
# ===========================================================================
STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&display=swap');

/* ── Reset & Base ──────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background-color: #F5F4F0 !important;
    font-family: 'DM Sans', sans-serif;
    color: #1A1A1A;
}

/* Remove padding padrão do Streamlit */
.main .block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── Sidebar ───────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(165deg, #0D3D38 0%, #0A2E2A 60%, #071F1C 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.06);
}

[data-testid="stSidebar"] * {
    color: #E8E4DC !important;
}

[data-testid="stSidebarContent"] {
    padding: 2rem 1.5rem !important;
}

/* Sidebar header */
.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 2rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.1);
}

.sidebar-logo .logo-mark {
    width: 36px;
    height: 36px;
    background: linear-gradient(135deg, #4ECDC4, #2BA99E);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    flex-shrink: 0;
}

.sidebar-logo .logo-text {
    line-height: 1.1;
}

.sidebar-logo .logo-main {
    font-family: 'DM Serif Display', serif;
    font-size: 1.1rem;
    font-weight: 400;
    color: #F0EBE1 !important;
    letter-spacing: 0.01em;
}

.sidebar-logo .logo-sub {
    font-size: 0.65rem;
    font-weight: 300;
    color: rgba(255,255,255,0.45) !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

/* Patient card na sidebar */
.patient-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 1.1rem;
    margin-bottom: 1.5rem;
}

.patient-card .patient-avatar {
    width: 44px;
    height: 44px;
    background: linear-gradient(135deg, #D4A843, #C49030);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    margin-bottom: 0.75rem;
}

.patient-card .patient-name {
    font-size: 0.9rem;
    font-weight: 600;
    color: #F0EBE1 !important;
    margin-bottom: 2px;
}

.patient-card .patient-id {
    font-size: 0.7rem;
    color: rgba(255,255,255,0.4) !important;
    font-family: 'DM Sans', monospace;
    letter-spacing: 0.05em;
}

.patient-card .patient-date {
    font-size: 0.72rem;
    color: rgba(255,255,255,0.4) !important;
    margin-top: 8px;
}

/* Status badge */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(78, 205, 196, 0.15);
    border: 1px solid rgba(78, 205, 196, 0.3);
    border-radius: 100px;
    padding: 3px 10px;
    font-size: 0.68rem;
    color: #4ECDC4 !important;
    margin-top: 10px;
    font-weight: 500;
    letter-spacing: 0.04em;
}

.status-dot {
    width: 6px;
    height: 6px;
    background: #4ECDC4;
    border-radius: 50%;
    animation: pulse-dot 2s infinite;
}

@keyframes pulse-dot {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* Sidebar section headers */
.sidebar-section-title {
    font-size: 0.62rem;
    font-weight: 600;
    color: rgba(255,255,255,0.3) !important;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin: 1.5rem 0 0.75rem 0;
}

/* Risk chips */
.risk-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px;
    margin-bottom: 1rem;
}

.risk-chip {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px;
    padding: 7px 9px;
    font-size: 0.68rem;
}

.risk-chip .risk-label {
    color: rgba(255,255,255,0.45) !important;
    margin-bottom: 2px;
    font-size: 0.62rem;
}

.risk-chip .risk-value {
    font-weight: 600;
    font-size: 0.75rem;
}

.risk-chip .risk-value.high { color: #FF6B6B !important; }
.risk-chip .risk-value.medium { color: #D4A843 !important; }
.risk-chip .risk-value.low { color: #4ECDC4 !important; }

/* Upload zone customizada */
.upload-hint {
    background: rgba(78, 205, 196, 0.08);
    border: 1px dashed rgba(78, 205, 196, 0.3);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    font-size: 0.75rem;
    color: rgba(255,255,255,0.5) !important;
    margin-top: 0.5rem;
}

/* ── Header Principal ───────────────────────────────────────────────────── */
.main-header {
    background: white;
    border-bottom: 1px solid #E8E4DC;
    padding: 1rem 2.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 100;
}

.header-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.15rem;
    color: #0D3D38;
    letter-spacing: -0.01em;
}

.header-subtitle {
    font-size: 0.72rem;
    color: #7A7A7A;
    margin-top: 1px;
    font-weight: 300;
}

/* ── Chat Container ─────────────────────────────────────────────────────── */
.chat-wrapper {
    max-width: 820px;
    margin: 0 auto;
    padding: 2rem 1.5rem 180px 1.5rem;
}

/* Mensagens */
.msg-row {
    display: flex;
    gap: 12px;
    margin-bottom: 1.75rem;
    align-items: flex-start;
    animation: msg-in 0.3s ease;
}

@keyframes msg-in {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

.msg-row.user {
    flex-direction: row-reverse;
}

/* Avatares */
.avatar {
    width: 38px;
    height: 38px;
    border-radius: 50%;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    position: relative;
    top: 2px;
}

.avatar.assistant-avatar {
    background: linear-gradient(135deg, #0D3D38, #1A5C55);
    box-shadow: 0 2px 12px rgba(13,61,56,0.25);
}

.avatar.user-avatar {
    background: linear-gradient(135deg, #D4A843, #C49030);
    box-shadow: 0 2px 12px rgba(212,168,67,0.25);
}

/* Balões */
.bubble-wrap {
    max-width: 82%;
}

.bubble {
    padding: 0.9rem 1.15rem;
    border-radius: 16px;
    font-size: 0.875rem;
    line-height: 1.65;
    color: #1A1A1A;
}

.bubble.assistant {
    background: white;
    border: 1px solid #E8E4DC;
    border-top-left-radius: 4px;
    box-shadow: 0 1px 8px rgba(0,0,0,0.05);
}

.bubble.user {
    background: #0D3D38;
    color: white;
    border-bottom-right-radius: 4px;
}

.bubble.user * { color: white !important; }

/* Timestamp */
.msg-time {
    font-size: 0.62rem;
    color: #ACACAC;
    margin-top: 5px;
    padding: 0 4px;
}

.msg-row.user .msg-time {
    text-align: right;
}

/* Nome do remetente */
.msg-sender {
    font-size: 0.7rem;
    font-weight: 600;
    margin-bottom: 4px;
    padding: 0 4px;
    color: #555;
}

.msg-row.user .msg-sender {
    text-align: right;
    color: #888;
}

/* Typing indicator */
.typing-indicator {
    display: flex;
    gap: 5px;
    padding: 14px 16px;
    background: white;
    border: 1px solid #E8E4DC;
    border-radius: 16px;
    border-top-left-radius: 4px;
    width: fit-content;
    box-shadow: 0 1px 8px rgba(0,0,0,0.05);
}

.typing-dot {
    width: 7px;
    height: 7px;
    background: #B0B0B0;
    border-radius: 50%;
    animation: typing 1.2s infinite;
}

.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }

@keyframes typing {
    0%, 100% { transform: translateY(0); opacity: 0.4; }
    50%       { transform: translateY(-5px); opacity: 1; }
}

/* ── Source Expander ────────────────────────────────────────────────────── */
.source-expander {
    max-width: 82%;
    margin-left: 50px;
    margin-top: -1rem;
    margin-bottom: 1.5rem;
}

.source-header {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.68rem;
    font-weight: 600;
    color: #0D3D38;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    cursor: pointer;
    padding: 6px 10px;
    background: rgba(13,61,56,0.04);
    border: 1px solid rgba(13,61,56,0.12);
    border-radius: 8px;
    width: fit-content;
}

.source-chunk {
    background: #FAFAF8;
    border: 1px solid #E8E4DC;
    border-left: 3px solid #4ECDC4;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    margin-top: 8px;
    font-size: 0.78rem;
    color: #3A3A3A;
    line-height: 1.55;
}

.source-chunk .source-meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
}

.source-tag {
    background: rgba(13,61,56,0.08);
    color: #0D3D38;
    border-radius: 4px;
    padding: 2px 7px;
    font-size: 0.63rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.source-score {
    font-size: 0.65rem;
    color: #9A9A9A;
    font-family: monospace;
}

/* ── Welcome Screen ─────────────────────────────────────────────────────── */
.welcome-container {
    text-align: center;
    padding: 4rem 2rem 3rem;
}

.welcome-icon {
    width: 72px;
    height: 72px;
    background: linear-gradient(135deg, #0D3D38, #1A5C55);
    border-radius: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2rem;
    margin: 0 auto 1.5rem;
    box-shadow: 0 8px 32px rgba(13,61,56,0.2);
}

.welcome-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.8rem;
    color: #0D3D38;
    margin-bottom: 0.5rem;
    letter-spacing: -0.02em;
}

.welcome-subtitle {
    font-size: 0.9rem;
    color: #6A6A6A;
    max-width: 460px;
    margin: 0 auto 2.5rem;
    line-height: 1.6;
    font-weight: 300;
}

.suggestion-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    max-width: 560px;
    margin: 0 auto;
}

.suggestion-pill {
    background: white;
    border: 1px solid #E0DDD5;
    border-radius: 12px;
    padding: 12px 16px;
    font-size: 0.8rem;
    color: #2A2A2A;
    cursor: pointer;
    transition: all 0.2s ease;
    text-align: left;
    line-height: 1.4;
}

.suggestion-pill:hover {
    border-color: #0D3D38;
    background: #F0F7F6;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(13,61,56,0.1);
}

.suggestion-pill .pill-emoji {
    font-size: 1rem;
    margin-bottom: 4px;
    display: block;
}

/* ── Disclaimer Footer ──────────────────────────────────────────────────── */
.disclaimer-footer {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: rgba(245,244,240,0.95);
    backdrop-filter: blur(8px);
    border-top: 1px solid #E0DDD5;
    padding: 10px 24px;
    z-index: 999;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
}

.disclaimer-text {
    font-size: 0.68rem;
    color: #8A8A8A;
    text-align: center;
    line-height: 1.4;
    max-width: 900px;
}

.disclaimer-icon {
    font-size: 0.9rem;
    flex-shrink: 0;
    opacity: 0.6;
}

/* ── Input Area ─────────────────────────────────────────────────────────── */
.input-dock {
    position: fixed;
    bottom: 44px;
    left: 0;
    right: 0;
    background: rgba(245,244,240,0.97);
    backdrop-filter: blur(12px);
    padding: 12px 24px 14px;
    border-top: 1px solid rgba(224,221,213,0.6);
    z-index: 998;
}

/* Streamlit input overrides */
[data-testid="stChatInput"] {
    max-width: 820px;
    margin: 0 auto;
}

[data-testid="stChatInput"] > div {
    border-radius: 14px !important;
    border: 1.5px solid #D5D1C8 !important;
    background: white !important;
    box-shadow: 0 2px 16px rgba(0,0,0,0.06) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}

[data-testid="stChatInput"] > div:focus-within {
    border-color: #0D3D38 !important;
    box-shadow: 0 2px 16px rgba(13,61,56,0.12) !important;
}

/* Streamlit expander override */
[data-testid="stExpander"] {
    border: 1px solid #E0DDD5 !important;
    border-radius: 10px !important;
    background: #FAFAF8 !important;
    box-shadow: none !important;
}

[data-testid="stExpander"] summary {
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    color: #0D3D38 !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: rgba(78, 205, 196, 0.05) !important;
    border: 1px dashed rgba(78, 205, 196, 0.4) !important;
    border-radius: 10px !important;
}

/* Streamlit button overrides */
.stButton > button {
    background: rgba(78, 205, 196, 0.15) !important;
    border: 1px solid rgba(78, 205, 196, 0.4) !important;
    color: #E8E4DC !important;
    border-radius: 8px !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
}

.stButton > button:hover {
    background: rgba(78, 205, 196, 0.25) !important;
    border-color: rgba(78, 205, 196, 0.6) !important;
}

/* Selectbox in sidebar */
[data-testid="stSelectbox"] {
    font-size: 0.8rem !important;
}

/* Hide Streamlit default elements — preserva controles da sidebar */
#MainMenu, footer { visibility: hidden; }
header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── Sidebar FIXA — sem colapso ────────────────────────────────────────── */

/* Força sidebar sempre expandida e visível */
[data-testid="stSidebar"] {
    transform: none !important;
    min-width: 280px !important;
    max-width: 320px !important;
    width: 300px !important;
    display: flex !important;
    visibility: visible !important;
    position: relative !important;
}

/* Esconde o botão de fechar/colapsar sidebar */
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"] {
    display: none !important;
    visibility: hidden !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #D5D1C8; border-radius: 10px; }

/* Sidebar markdown override */
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown span {
    font-size: 0.8rem !important;
}
</style>
"""


# ===========================================================================
# DADOS MOCK — Perfis Genômicos de Demonstração
# ===========================================================================

MOCK_PROFILES = {
    "Carlos Silva (PAC-2024-001)": {
        "id": "PAC-2024-001",
        "nome": "Carlos Silva",
        "data_exame": "20/11/2024",
        "riscos": {
            "Diabetes T2": ("38%", "high"),
            "Cardiovascular": ("24%", "medium"),
            "Alzheimer": ("19%", "medium"),
            "Câncer Mama": ("12%", "low"),
        },
        "ancestralidade": "68% Europeia · 22% Africana · 10% Ameríndia",
        "chunks": {
            "diabetes": {
                "section": "riscos_doencas",
                "subsection": "diabetes_tipo_2",
                "text": "Risco genético para Diabetes Tipo 2: Nível de risco: Elevado. Risco individual: 38%. Média populacional: 11%. Variantes identificadas: rs7903146, rs12255372, rs9939609. Genes envolvidos: TCF7L2, FTO, PPARG. Risco 3.5x acima da média populacional. Fatores de estilo de vida relevantes: Dieta com baixo índice glicêmico é altamente recomendada; Exercício aeróbico 150 min/semana reduz o risco em até 58%.",
                "score": 0.94,
            },
            "cardiovascular": {
                "section": "riscos_doencas",
                "subsection": "doenca_cardiovascular",
                "text": "Risco genético para Doença Cardiovascular: Nível de risco: Moderado. Risco individual: 24%. Média populacional: 15%. Variantes identificadas: rs1333049, rs2383207, rs10757278. Genes envolvidos: CDKN2B-AS1, CXCL12. Risco 1.6x acima da média. Responde bem a estatinas.",
                "score": 0.81,
            },
            "wellbeing_caffeine": {
                "section": "bem_estar",
                "subsection": "metabolismo_cafeina",
                "text": "Bem-estar genômico – nutricao: Metabolismo de Cafeína. Resultado genético: Metabolizador Lento. CYP1A2 *1A/*1F. Impacto: Cafeína permanece mais tempo no organismo. Consumo acima de 200mg/dia associado a maior risco cardiovascular neste genótipo. Recomendação personalizada: Limitar a 1-2 xícaras de café por dia. Evitar cafeína após 14h.",
                "score": 0.77,
            },
            "farmacogenomica": {
                "section": "bem_estar",
                "subsection": "farmacogenomica",
                "text": "Bem-estar genômico – farmacogenomica: Metabolismo de Varfarina. Resultado genético: Metabolizador Intermediário. CYP2C9 *1/*2 + VKORC1 AG. Necessita de dose reduzida de varfarina. Risco aumentado de sangramentos com dose padrão. Estatinas: Polimorfismo SLCO1B1 rs4149056 CT associado a maior risco de miopatia. Preferir rosuvastatina ou pravastatina.",
                "score": 0.72,
            },
            "ancestralidade": {
                "section": "ancestralidade",
                "subsection": "composicao",
                "text": "Ancestralidade Europeia: 68% de composição genética. Haplogrupo: H1a1. Ancestralidade Africana: 22%. Haplogrupo: L3b1. Proteção parcial contra malária. Ancestralidade Ameríndia: 10%. Haplogrupo: A2. Implicações clínicas: Predisposição aumentada para diabetes tipo 2 e síndrome metabólica.",
                "score": 0.68,
            },
            "recomendacoes": {
                "section": "recomendacoes",
                "subsection": "prioridade_alta",
                "text": "Recomendação clínica – Prevenção Metabólica: Consultar endocrinologista para estratégia preventiva de diabetes. Solicitar TOTG, hemoglobina glicada e insulina de jejum. Prioridade: Alta. Especialista indicado: Endocrinologista. Alerta Farmacogenômico: Informar médico sobre resultados de varfarina e estatinas em toda consulta.",
                "score": 0.65,
            },
        },
    }
}

# ===========================================================================
# MOTOR DE RESPOSTAS MOCK (substitui o RAG real em demo)
# ===========================================================================

MOCK_RESPONSES: list[dict] = [
    {
        "keywords": ["diabetes", "glicose", "açúcar", "insulina", "glicemia"],
        "answer": """**Seu risco genético para Diabetes Tipo 2 é Elevado — 38% vs. 11% da média populacional.**

Foram identificadas 3 variantes de risco nos genes **TCF7L2**, **FTO** e **PPARG** — os principais genes associados ao DM2 conhecidos até hoje. O gene TCF7L2, em especial, é o fator genético de maior impacto para diabetes tipo 2.

**O que isso significa na prática?**
Seu genoma indica uma predisposição 3,5× acima da média, mas importante ressaltar: **genética não é destino**. Estudos mostram que mudanças de estilo de vida podem reduzir o risco em até 58%, independente da herança genética.

**Ações recomendadas:**
- 🥗 Dieta com baixo índice glicêmico (priorizar fibras, reduzir ultraprocessados)
- 🏃 Exercício aeróbico: 150 min/semana (caminhada, natação, ciclismo)
- 🩸 Exames anuais: TOTG, hemoglobina glicada e insulina de jejum
- 👨‍⚕️ Consulta com endocrinologista para estratégia preventiva personalizada

Gostaria de saber mais sobre algum desses aspectos?""",
        "chunk_keys": ["diabetes", "ancestralidade", "recomendacoes"],
    },
    {
        "keywords": ["coração", "cardiovascular", "infarto", "pressão", "colesterol", "estatina"],
        "answer": """**Seu risco cardiovascular é Moderado — 24% vs. 15% da média populacional.**

Foram identificadas variantes nos genes **CDKN2B-AS1** e **CXCL12**, localizados no locus cromossômico 9p21, associados a maior risco de infarto agudo do miocárdio e angina instável.

**Farmacogenômica importante:** caso você precise usar estatinas no futuro, seu perfil (SLCO1B1 rs4149056 CT) indica **preferência por rosuvastatina ou pravastatina** — há risco intermediário de miopatia com sinvastatina em doses elevadas. Mencione isso ao seu médico.

**Ações recomendadas:**
- 🫀 Avaliação cardiológica com lipidograma completo e PCR ultra-sensível
- 🫒 Dieta mediterrânea tem benefício documentado para este genótipo
- 📊 Monitorar pressão arterial regularmente
- 💊 Se prescrito estatina, informe seu médico sobre o resultado farmacogenômico

Quer saber mais sobre a interação com estatinas?""",
        "chunk_keys": ["cardiovascular", "farmacogenomica", "recomendacoes"],
    },
    {
        "keywords": ["café", "cafeína", "estimulante", "energia"],
        "answer": """**Você é um Metabolizador Lento de cafeína** (genótipo CYP1A2 *1A/*1F).

Isso significa que a cafeína permanece no seu organismo por mais tempo do que na média das pessoas. Neste genótipo específico, consumo acima de **200mg/dia** (equivalente a ~2 xícaras de café) está associado a maior risco cardiovascular.

**Na prática:**
- ☕ Limite a 1–2 xícaras de café por dia
- ⏰ Evite cafeína após 14h (pode interferir no sono)
- 🍵 Chás verdes e chás pretos também contêm cafeína — moderação igualmente recomendada
- 🧠 Isso se conecta ao seu risco cardiovascular moderado: metabolizadores lentos que consomem muito café têm risco elevado

Há algo mais que gostaria de saber sobre sua nutrição personalizada?""",
        "chunk_keys": ["wellbeing_caffeine", "cardiovascular"],
    },
    {
        "keywords": ["vitamina", "lactose", "nutrição", "leite", "suplemento", "folato", "mthfr"],
        "answer": """Seu perfil genômico revela **3 aspectos nutricionais importantes**:

**1. Intolerância à Lactose** (LCT -13910 CC)
Você possui o genótipo associado à não-persistência da lactase na vida adulta. Cerca de 70% dos portadores apresentam sintomas. Prefira laticínios fermentados (iogurte, kefir) ou use suplemento de lactase.

**2. Absorção Reduzida de Vitamina D** (VDR rs2228570 CT)
Polimorfismo no receptor de vitamina D impacta a absorção intestinal. Recomendação: checar 25-OH Vitamina D semestralmente e considerar suplementação conforme exame.

**3. MTHFR Heterozigoto** (C677T — CT)
Atividade da enzima reduzida em ~35%. Se precisar suplementar ácido fólico, prefira **folato na forma 5-MTHF** (ativo) ao invés do ácido fólico sintético convencional.

Sugestão: consultar nutricionista especializado em **nutrigenômica** para um plano alimentar integrado a esses resultados.

Posso detalhar algum desses pontos?""",
        "chunk_keys": ["wellbeing_caffeine", "recomendacoes"],
    },
    {
        "keywords": ["ancestral", "origem", "herança", "étni", "african", "europe", "amerínd"],
        "answer": """**Sua composição ancestral genética:**

| Origem | Proporção | Haplogrupo |
|--------|-----------|------------|
| Europeia | **68%** | H1a1 |
| Africana | **22%** | L3b1 |
| Ameríndia | **10%** | A2 |

**Implicações clínicas relevantes:**
- A **ancestralidade Ameríndia** (10%) contribui para a predisposição aumentada a diabetes tipo 2 e síndrome metabólica — populações com essa herança apresentam maior sensibilidade à dieta ocidental
- A **ancestralidade Africana** (22%) traz proteção parcial contra malária e os marcadores de traço falciforme foram testados como **negativos** — sem preocupação nesse aspecto
- A **ancestralidade Europeia** dominante (68%) eleva a atenção para fenilcetonúria e hemocromatose — ambas descartadas no seu painel

Lembre que ancestralidade genética é um dado populacional; seu perfil individual é sempre uma combinação única dessas influências.

Quer explorar como sua ancestralidade se conecta a algum risco específico?""",
        "chunk_keys": ["ancestralidade", "diabetes", "recomendacoes"],
    },
    {
        "keywords": ["remédio", "medicamento", "varfarina", "anticoagulante", "farmaco", "droga"],
        "answer": """**Seu perfil farmacogenômico tem dois alertas importantes para consultas médicas:**

**1. Varfarina / Anticoagulantes** (CYP2C9 *1/*2 + VKORC1 AG)
Você é um **metabolizador intermediário** de varfarina. Com a dose padrão, há risco aumentado de sangramentos. Se prescrito, a dose inicial recomendada é **40–60% da dose padrão** — o médico deve ajustar com base nesse resultado.

**2. Estatinas** (SLCO1B1 rs4149056 CT)
Há risco intermediário de **miopatia** (dor muscular) com certas estatinas. Preferência: **rosuvastatina ou pravastatina**. Evite sinvastatina 80mg. Monitorar CPK nos primeiros 6 meses de uso.

⚠️ **Recomendação:** Salve este resultado no seu celular e apresente ao médico em **toda nova consulta ou prescrição**, especialmente cirurgias, hospitalização e tratamentos crônicos.

Quer saber mais sobre algum medicamento específico?""",
        "chunk_keys": ["farmacogenomica", "cardiovascular", "recomendacoes"],
    },
    {
        "keywords": ["exercício", "treino", "músculo", "esporte", "atividade física", "fitness"],
        "answer": """**Seu perfil de aptidão física genética:**

**Tipo Muscular:** Você possui o genótipo **ACTN3 R577X RR** — associado à **predominância de fibras de contração rápida** (força e potência). Isso significa que seu genoma favorece resposta superior a:
- 💪 Musculação e treino de força
- ⚡ HIIT (High Intensity Interval Training)
- 🏃 Sprints e exercícios explosivos

**Recuperação Muscular:** Genótipo IL-6 GC indica resposta inflamatória moderada ao exercício — recuperação dentro da média. Recomendado: 48–72h entre sessões intensas.

**Integração com seus riscos:** Dado seu risco elevado para diabetes, o exercício aeróbico (150min/semana) é altamente recomendado. Uma boa estratégia seria combinar **treino de força 3x/semana** com **caminhadas ou ciclismo** — potencializa o benefício metabólico.

Gostaria de entender como conciliar treino com a prevenção do diabetes?""",
        "chunk_keys": ["wellbeing_caffeine", "diabetes"],
    },
]

DEFAULT_RESPONSE = {
    "answer": """Obrigado pela pergunta! Consultei seu perfil genômico mas **não encontrei informações específicas** sobre esse tema no relatório carregado.

Algumas sugestões:
- Tente reformular a pergunta mencionando a condição, gene ou tema específico
- Consulte diretamente com seu **médico geneticista ou especialista** para essa dúvida
- Você também pode perguntar sobre: diabetes, risco cardiovascular, ancestralidade, medicamentos, nutrição, cafeína ou atividade física

Posso ajudar com algum desses tópicos?""",
    "chunk_keys": [],
}


# ===========================================================================
# FUNÇÕES DO MOTOR MOCK
# ===========================================================================

def mock_rag_answer(question: str, profile: dict) -> dict[str, Any]:
    """
    Simula o pipeline RAG retornando resposta + chunks de fonte.
    Em produção: substituir por GenomicRAGPipeline.answer(question)
    """
    q_lower = question.lower()
    matched = None

    for resp in MOCK_RESPONSES:
        if any(kw in q_lower for kw in resp["keywords"]):
            matched = resp
            break

    if not matched:
        return {
            "answer": DEFAULT_RESPONSE["answer"],
            "sources": [],
        }

    # Monta os chunks de fonte
    sources = []
    for key in matched["chunk_keys"]:
        if key in profile["chunks"]:
            sources.append(profile["chunks"][key])

    return {
        "answer": matched["answer"],
        "sources": sources,
    }


def simulate_typing(container, text: str, delay: float = 0.012):
    """Simula efeito de digitação streaming."""
    placeholder = container.empty()
    displayed = ""
    for char in text:
        displayed += char
        placeholder.markdown(displayed + "▌")
        time.sleep(delay)
    placeholder.markdown(displayed)
    return displayed


# ===========================================================================
# COMPONENTES DE UI
# ===========================================================================

def render_sidebar():
    """Renderiza a barra lateral completa."""
    with st.sidebar:
        # Logo
        st.markdown("""
        <div class="sidebar-logo">
            <div class="logo-mark">🧬</div>
            <div class="logo-text">
                <div class="logo-main">Genera Intelligence</div>
                <div class="logo-sub">Genomic AI · Dasa</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Upload de relatório (sempre visível no topo) ──────────────────
        st.markdown('<div class="sidebar-section-title">Carregar Relatório</div>',
                    unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "JSON ou PDF do relatório Genera",
            type=["json", "pdf"],
            label_visibility="collapsed",
            help="Carregue o JSON ou PDF gerado pelo pipeline Genera",
        )

        if uploaded and uploaded.name != st.session_state.uploaded_filename:
            # Novo arquivo: processa e indexa
            try:
                with st.spinner("Indexando perfil genômico..."):
                    time.sleep(1.2)  # Simula pipeline RAG
                    if uploaded.name.lower().endswith(".pdf"):
                        profile = extract_profile_from_pdf(uploaded.read())
                    else:
                        uploaded.seek(0)
                        raw_json = json.load(uploaded)
                        profile = extract_profile_from_json(raw_json)
                st.session_state.current_profile = profile
                st.session_state.uploaded_filename = uploaded.name
                st.session_state.messages = []
                st.session_state.show_welcome = True
                st.success(f"✓ {uploaded.name} carregado — {profile['nome']}")
                st.rerun()
            except Exception as e:
                st.error(f"Erro ao processar arquivo: {e}")

        # ── Perfil Carregado ──────────────────────────────────────────────
        st.markdown('<div class="sidebar-section-title">Perfil Carregado</div>',
                    unsafe_allow_html=True)

        profile = st.session_state.current_profile

        if profile is None:
            # Nenhum arquivo carregado ainda — estado vazio
            st.markdown("""
            <div style="background: rgba(255,255,255,0.04); border: 1px dashed rgba(255,255,255,0.12);
                        border-radius: 12px; padding: 1.2rem; text-align: center;
                        font-size: 0.75rem; color: rgba(255,255,255,0.35); line-height: 1.6;">
                Nenhum laudo carregado.<br>Faça o upload do JSON acima.
            </div>
            """, unsafe_allow_html=True)
        else:
            # Card do paciente preenchido com dados do JSON
            st.markdown(f"""
            <div class="patient-card">
                <div class="patient-avatar">👤</div>
                <div class="patient-name">{profile['nome']}</div>
                <div class="patient-id">{profile['id']}</div>
                <div class="patient-date">📅 Exame: {profile['data_exame']}</div>
                <div class="status-badge">
                    <div class="status-dot"></div>
                    Relatório Ativo
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Riscos extraídos do JSON ──────────────────────────────────
            if profile.get("riscos"):
                st.markdown('<div class="sidebar-section-title">Principais Riscos</div>',
                            unsafe_allow_html=True)

                risk_html = '<div class="risk-grid">'
                for disease, (pct, level) in profile["riscos"].items():
                    risk_html += f"""
                    <div class="risk-chip">
                        <div class="risk-label">{disease}</div>
                        <div class="risk-value {level}">{pct}</div>
                    </div>"""
                risk_html += "</div>"
                st.markdown(risk_html, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="font-size: 0.72rem; color: rgba(255,255,255,0.35); margin-bottom: 1rem;">
                    Nenhum risco encontrado no JSON.
                </div>
                """, unsafe_allow_html=True)

        # ── Nova conversa ─────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄  Nova Conversa", use_container_width=True):
            st.session_state.messages = []
            st.session_state.show_welcome = True
            st.rerun()

        # Versão
        st.markdown("""
        <div style="margin-top: 2rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.08);
                    font-size: 0.62rem; color: rgba(255,255,255,0.2); text-align: center;">
            Genera Intelligence v2.0 · Sprint 2<br>
            Challenge Dasa/Genera 2024
        </div>
        """, unsafe_allow_html=True)


def render_welcome():
    """Tela de boas-vindas quando não há mensagens."""
    profile = st.session_state.get("current_profile")
    first_name = profile["nome"].split()[0] if profile else None
    greeting = f"Olá, {first_name}!" if first_name else "Olá, bem-vindo!"
    subtitle = (
        f"Seu perfil genômico de <strong>{profile['nome']}</strong> está carregado. "
        "Faça perguntas sobre seus riscos de saúde, ancestralidade, nutrição e "
        "farmacogenômica — tudo baseado exclusivamente no seu relatório genético."
        if profile else
        "Converse com seu perfil genômico personalizado. Faça perguntas sobre "
        "seus riscos de saúde, ancestralidade, nutrição e farmacogenômica — "
        "tudo baseado exclusivamente no seu relatório genético."
    )
    st.markdown(f"""
    <div class="welcome-container">
        <div class="welcome-icon">🧬</div>
        <div class="welcome-title">{greeting}</div>
        <div class="welcome-subtitle">{subtitle}</div>
        <div class="suggestion-grid">
            <div class="suggestion-pill" onclick="void(0)">
                <span class="pill-emoji">🩺</span>
                Qual meu risco de desenvolver diabetes?
            </div>
            <div class="suggestion-pill">
                <span class="pill-emoji">❤️</span>
                Como está minha saúde cardiovascular?
            </div>
            <div class="suggestion-pill">
                <span class="pill-emoji">☕</span>
                Posso tomar muito café com meu genótipo?
            </div>
            <div class="suggestion-pill">
                <span class="pill-emoji">💊</span>
                Tenho algum alerta sobre medicamentos?
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_message(role: str, content: str, timestamp: str,
                   sources: list | None = None, msg_index: int = 0):
    """Renderiza uma mensagem no chat com avatar e balão."""

    if role == "user":
        profile = st.session_state.get("current_profile")
        sender_name = profile["nome"].split()[0] if profile else "Você"
        st.markdown(f"""
        <div class="msg-row user">
            <div class="avatar user-avatar">👤</div>
            <div class="bubble-wrap">
                <div class="msg-sender">{sender_name}</div>
                <div class="bubble user">{content}</div>
                <div class="msg-time">{timestamp}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:  # assistant
        st.markdown(f"""
        <div class="msg-row">
            <div class="avatar assistant-avatar">🧬</div>
            <div class="bubble-wrap">
                <div class="msg-sender">Assistente Genera</div>
                <div class="bubble assistant">{content}</div>
                <div class="msg-time">{timestamp}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Expander de fontes
        if sources:
            with st.expander(f"📎 {len(sources)} trecho(s) do relatório utilizados como fonte"):
                for i, src in enumerate(sources, 1):
                    section_label = src.get("section", "").replace("_", " ").title()
                    subsection = src.get("subsection", "").replace("_", " ").title()
                    score = src.get("score", 0.0)
                    text = src.get("text", "")

                    st.markdown(f"""
                    <div class="source-chunk">
                        <div class="source-meta">
                            <span class="source-tag">
                                {section_label} › {subsection}
                            </span>
                            <span class="source-score">
                                similaridade: {score:.2f}
                            </span>
                        </div>
                        <div>{text}</div>
                    </div>
                    """, unsafe_allow_html=True)


def render_disclaimer():
    """Footer de disclaimer obrigatório — fixo na parte inferior."""
    st.markdown("""
    <div class="disclaimer-footer">
        <span class="disclaimer-icon">⚕️</span>
        <div class="disclaimer-text">
            <strong>Aviso Importante:</strong> Esta ferramenta é de caráter <strong>exclusivamente informativo</strong> 
            e baseada em dados genômicos. As informações apresentadas <strong>não substituem</strong> consulta, 
            diagnóstico ou prescrição médica. Sempre consulte um profissional de saúde habilitado antes de tomar 
            qualquer decisão clínica. · Genera Intelligence · Dasa S.A. · Uso restrito ao titular do relatório.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ===========================================================================
# ESTADO DA SESSÃO
# ===========================================================================

def extract_profile_from_pdf(pdf_bytes: bytes) -> dict:
    """
    Extrai dados do paciente de um PDF de relatório genômico.
    Faz parsing de texto livre buscando nome, ID, data e riscos.
    """
    try:
        import io
        # tenta pypdf (novo nome) ou PyPDF2 (legado)
        try:
            from pypdf import PdfReader
        except ImportError:
            from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        text = ""

    # ── Extrai campos básicos ──────────────────────────────────────────────
    nome = "Paciente"
    pid = "—"
    data_exame = "—"

    m = re.search(r"(?:paciente|nome)[:\s]+([A-ZÀ-Ú][a-zà-ú]+(?: [A-ZÀ-Ú][a-zà-ú]+)+)", text, re.I)
    if m:
        nome = m.group(1).strip()

    m = re.search(r"(?:id|código|codigo|prontuário)[:\s]*([\w\-]+)", text, re.I)
    if m:
        pid = m.group(1).strip()

    m = re.search(r"(?:data.*?exame|exame.*?data)[:\s]*([\d]{2}[\/\-\.][\d]{2}[\/\-\.][\d]{2,4})", text, re.I)
    if m:
        data_exame = m.group(1).strip()

    # ── Extrai riscos ──────────────────────────────────────────────────────
    RISK_KEYWORDS = {
        "Diabetes T2": ["diabetes", "dm2", "diabetes tipo 2"],
        "Cardiovascular": ["cardiovascular", "infarto", "doença cardíaca"],
        "Alzheimer": ["alzheimer"],
        "Câncer Mama": ["câncer de mama", "cancer de mama", "mama"],
        "Trombose": ["trombose", "tvp"],
    }
    LEVEL_MAP = {
        "elevado": "high", "alto": "high",
        "moderado": "medium", "médio": "medium", "medio": "medium",
        "baixo": "low",
    }

    riscos: dict = {}
    for label, kws in RISK_KEYWORDS.items():
        for kw in kws:
            pattern = rf"{kw}.{{0,120}}?(\d{{1,3}})\s*%"
            m = re.search(pattern, text, re.I | re.S)
            if m:
                pct = int(m.group(1))
                # tenta detectar nível pelo texto próximo
                snippet = text[max(0, m.start()-80):m.end()+80].lower()
                level = "medium"
                for word, cls in LEVEL_MAP.items():
                    if word in snippet:
                        level = cls
                        break
                riscos[label] = (f"{pct}%", level)
                break

    chunks = MOCK_PROFILES.get("Carlos Silva (PAC-2024-001)", {}).get("chunks", {})

    return {
        "id": pid,
        "nome": nome,
        "data_exame": data_exame,
        "riscos": riscos,
        "chunks": chunks,
        "_raw": {"fonte": "pdf", "texto_extraido": text[:2000]},
    }


def extract_profile_from_json(data: dict) -> dict:
    """
    Extrai nome do paciente e riscos diretamente do JSON da Sprint 1.
    Classifica o nível de risco comparando percentual individual vs. média populacional.
    """
    patient = data.get("paciente", {})
    nome = patient.get("nome", "Paciente")
    pid = patient.get("id", "—")
    data_exame = patient.get("data_exame", patient.get("data_nascimento", "—"))

    # Extrai riscos de doenças do JSON
    riscos_raw = data.get("riscos_doencas", [])
    if isinstance(riscos_raw, dict):
        riscos_raw = riscos_raw.get("doencas", list(riscos_raw.values()))

    riscos: dict[str, tuple[str, str]] = {}
    for item in riscos_raw[:6]:  # máx 6 chips na sidebar
        if not isinstance(item, dict):
            continue
        disease_name = item.get("nome", item.get("doenca", ""))
        if not disease_name:
            continue

        pct_individual = item.get("percentual_risco", item.get("risco"))
        pct_pop = item.get("media_populacional")
        nivel = item.get("nivel_risco", item.get("risco", "")).lower()

        # Nível por texto se disponível
        if "elevado" in nivel or "alto" in nivel:
            level_class = "high"
        elif "moderado" in nivel or "médio" in nivel or "medio" in nivel:
            level_class = "medium"
        elif "baixo" in nivel:
            level_class = "low"
        # Fallback: calcula pela razão individual/populacional
        elif pct_individual and pct_pop:
            try:
                ratio = float(str(pct_individual).replace("%", "")) / float(str(pct_pop).replace("%", ""))
                level_class = "high" if ratio >= 2.5 else ("medium" if ratio >= 1.3 else "low")
            except (ValueError, ZeroDivisionError):
                level_class = "medium"
        else:
            level_class = "medium"

        display_pct = f"{pct_individual}%" if pct_individual and "%" not in str(pct_individual) else str(pct_individual or "—")
        # Abrevia nome longo para caber no chip
        short_name = disease_name if len(disease_name) <= 16 else disease_name[:14] + "…"
        riscos[short_name] = (display_pct, level_class)

    # Mantém chunks do mock se o perfil carregado for o de demonstração
    chunks = MOCK_PROFILES.get("Carlos Silva (PAC-2024-001)", {}).get("chunks", {})

    return {
        "id": pid,
        "nome": nome,
        "data_exame": data_exame,
        "riscos": riscos,
        "chunks": chunks,
        "_raw": data,
    }


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "show_welcome" not in st.session_state:
        st.session_state.show_welcome = True
    if "current_profile" not in st.session_state:
        st.session_state.current_profile = None  # Nenhum perfil carregado até upload
    if "uploaded_filename" not in st.session_state:
        st.session_state.uploaded_filename = None


# ===========================================================================
# APP PRINCIPAL
# ===========================================================================

def main():
    # Injeta estilos
    st.markdown(STYLES, unsafe_allow_html=True)

    # Inicializa estado
    init_session_state()

    # Sidebar
    render_sidebar()

    # Header
    st.markdown("""
    <div class="main-header">
        <div>
            <div class="header-title">🧬 Chat Genômico Personalizado</div>
            <div class="header-subtitle">Perguntas baseadas exclusivamente no seu relatório Genera</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Container principal do chat
    st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)

    # Tela de boas-vindas ou histórico
    if st.session_state.show_welcome and not st.session_state.messages:
        render_welcome()
    else:
        for i, msg in enumerate(st.session_state.messages):
            render_message(
                role=msg["role"],
                content=msg["content"],
                timestamp=msg["timestamp"],
                sources=msg.get("sources"),
                msg_index=i,
            )

    st.markdown('</div>', unsafe_allow_html=True)

    # Disclaimer fixo
    render_disclaimer()

    # ── Input do Chat ────────────────────────────────────────────────────
    no_profile = st.session_state.current_profile is None
    input_placeholder = (
        "Carregue um relatório JSON para começar..."
        if no_profile
        else "Faça uma pergunta sobre seu perfil genômico..."
    )

    if prompt := st.chat_input(input_placeholder, key="chat_input", disabled=no_profile):
        now = datetime.now().strftime("%H:%M")

        # Esconde welcome
        st.session_state.show_welcome = False

        # Adiciona mensagem do usuário
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": now,
            "sources": None,
        })

        # Gera resposta
        profile = st.session_state.current_profile
        result = mock_rag_answer(prompt, profile)

        # Simula delay de processamento
        with st.spinner("Consultando seu perfil genômico..."):
            processing_time = random.uniform(0.8, 1.6)
            time.sleep(processing_time)

        # Adiciona resposta do assistente
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "timestamp": datetime.now().strftime("%H:%M"),
            "sources": result["sources"],
        })

        st.rerun()


if __name__ == "__main__":
    main()
