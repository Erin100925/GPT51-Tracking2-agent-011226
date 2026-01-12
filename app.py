import os
import json
import re
import random
from typing import Dict, Any, List, Optional

import streamlit as st
import yaml

# External LLM SDKs
import openai
import google.generativeai as genai
import anthropic


# ---------------------------
# Constants & UI Dictionaries
# ---------------------------

DEFAULT_MAX_TOKENS = 12000

SUPPORTED_MODELS = [
    "gpt-4o-mini",
    "gpt-4.1-mini",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3-pro-preview",
    "claude-3-5-sonnet-latest",
    "claude-3-5-haiku-latest",
    "grok-4-fast-reasoning",
    "grok-3-mini",
]

PAINTER_STYLES = [
    "Vincent van Gogh",
    "Claude Monet",
    "Pablo Picasso",
    "Leonardo da Vinci",
    "Salvador Dalí",
    "Frida Kahlo",
    "Edvard Munch",
    "Gustav Klimt",
    "Georgia O’Keeffe",
    "Jackson Pollock",
    "Henri Matisse",
    "Wassily Kandinsky",
    "Paul Cézanne",
    "Joan Miró",
    "Rembrandt",
    "Caravaggio",
    "Diego Velázquez",
    "Marc Chagall",
    "Roy Lichtenstein",
    "Andy Warhol",
]

PAINTER_STYLE_PALETTES = {
    "Vincent van Gogh": ("linear-gradient(135deg,#0f172a,#1e3a8a)", "#fbbf24"),
    "Claude Monet": ("linear-gradient(135deg,#e0f2fe,#bae6fd)", "#0369a1"),
    "Pablo Picasso": ("linear-gradient(135deg,#111827,#4b5563)", "#f97316"),
    "Leonardo da Vinci": ("linear-gradient(135deg,#fef3c7,#fde68a)", "#92400e"),
    "Salvador Dalí": ("linear-gradient(135deg,#fef2f2,#fee2e2)", "#b91c1c"),
    "Frida Kahlo": ("linear-gradient(135deg,#fdf2f8,#fce7f3)", "#be123c"),
    "Edvard Munch": ("linear-gradient(135deg,#111827,#7f1d1d)", "#f97316"),
    "Gustav Klimt": ("linear-gradient(135deg,#fef3c7,#facc15)", "#b45309"),
    "Georgia O’Keeffe": ("linear-gradient(135deg,#dcfce7,#bbf7d0)", "#166534"),
    "Jackson Pollock": ("linear-gradient(135deg,#020617,#1f2937)", "#67e8f9"),
    "Henri Matisse": ("linear-gradient(135deg,#eff6ff,#dbeafe)", "#1d4ed8"),
    "Wassily Kandinsky": ("linear-gradient(135deg,#f9fafb,#e5e7eb)", "#7c3aed"),
    "Paul Cézanne": ("linear-gradient(135deg,#fef9c3,#fef08a)", "#ca8a04"),
    "Joan Miró": ("linear-gradient(135deg,#faf5ff,#ede9fe)", "#7e22ce"),
    "Rembrandt": ("linear-gradient(135deg,#0f172a,#1f2937)", "#facc15"),
    "Caravaggio": ("linear-gradient(135deg,#020617,#1f2937)", "#f97316"),
    "Diego Velázquez": ("linear-gradient(135deg,#111827,#374151)", "#facc15"),
    "Marc Chagall": ("linear-gradient(135deg,#eef2ff,#e0e7ff)", "#4c1d95"),
    "Roy Lichtenstein": ("linear-gradient(135deg,#faf5ff,#fee2e2)", "#1d4ed8"),
    "Andy Warhol": ("linear-gradient(135deg,#ecfeff,#e0f2fe)", "#e11d48"),
}

LANG_EN = "en"
LANG_ZH = "zh-TW"

UI_TEXT = {
    LANG_EN: {
        "title": "Agentic AI Project Orchestrator",
        "project_input": "Project Description / Tender Text",
        "run_orchestrator": "Generate Project Plan",
        "orchestrator_settings": "Orchestrator Settings",
        "model": "Model",
        "max_tokens": "Max tokens",
        "system_prompt": "Orchestrator System Prompt (optional, advanced)",
        "dashboard": "Dashboard",
        "work_breakdown": "Work Breakdown",
        "timeline": "Timeline",
        "agent_matrix": "Agent Allocation",
        "risk_heatmap": "Risk Heatmap",
        "dependencies": "Dependency Graph",
        "config": "Configuration",
        "agents_tab": "Agents & Execution",
        "skills_tab": "Skills",
        "chat_tab": "Refinement / Prompt on Results",
        "theme": "Theme",
        "light": "Light",
        "dark": "Dark",
        "language": "Language",
        "painter_style": "Painter Style",
        "jackpot": "Jackpot!",
        "api_section": "API Keys",
        "api_hint": "If environment variables exist, they will be used. You only need to fill missing keys.",
        "openai_key": "OpenAI API Key",
        "gemini_key": "Gemini API Key",
        "anthropic_key": "Anthropic API Key",
        "grok_key": "Grok API Key",
        "save_keys": "Save keys to session",
        "plan_missing": "No project plan yet. Please generate it first.",
        "run_agent": "Run this agent",
        "agent_input": "Agent input / task context",
        "agent_result_view": "Result view",
        "text_view": "Plain text",
        "markdown_view": "Markdown",
        "shared_handoff": "Shared Agent Handoff Buffer",
        "use_last_output": "Use last agent output in handoff buffer",
        "refresh_config": "Reload agents.yaml & SKILL.md",
        "wow_status": "WOW Status",
        "wow_agents": "Agents loaded",
        "wow_workitems": "Work items",
        "wow_risks": "Identified risks",
        "wow_ready": "Ready to orchestrate",
        "chat_prompt": "Refinement prompt (Prompt on Results)",
        "run_refinement": "Run refinement",
        "apply_refinement": "Apply refined fragment to plan",
        "nodes_label": "Nodes are work items; arrows show dependencies.",
    },
    LANG_ZH: {
        "title": "智慧代理專案協調器",
        "project_input": "專案說明 / 標案文字",
        "run_orchestrator": "產生專案計畫",
        "orchestrator_settings": "協調器設定",
        "model": "模型",
        "max_tokens": "最大 token 數",
        "system_prompt": "協調器系統提示（選填，高階設定）",
        "dashboard": "儀表板",
        "work_breakdown": "工作分解結構",
        "timeline": "時程規劃",
        "agent_matrix": "代理與資源配置",
        "risk_heatmap": "風險熱度圖",
        "dependencies": "相依關係圖",
        "config": "設定",
        "agents_tab": "代理與執行",
        "skills_tab": "技能",
        "chat_tab": "優化 / 結果再提示",
        "theme": "主題",
        "light": "亮色",
        "dark": "暗色",
        "language": "語言",
        "painter_style": "畫家風格",
        "jackpot": "隨機大補帖",
        "api_section": "API 金鑰",
        "api_hint": "若已設定環境變數，將自動使用。僅需填寫缺少的金鑰即可。",
        "openai_key": "OpenAI API 金鑰",
        "gemini_key": "Gemini API 金鑰",
        "anthropic_key": "Anthropic API 金鑰",
        "grok_key": "Grok API 金鑰",
        "save_keys": "儲存金鑰到本次工作階段",
        "plan_missing": "目前尚未有專案計畫，請先執行協調器。",
        "run_agent": "執行此代理",
        "agent_input": "代理輸入 / 任務內容",
        "agent_result_view": "結果檢視模式",
        "text_view": "純文字",
        "markdown_view": "Markdown",
        "shared_handoff": "代理交辦共用緩衝區",
        "use_last_output": "以上一個代理輸出更新交辦內容",
        "refresh_config": "重新載入 agents.yaml 與 SKILL.md",
        "wow_status": "WOW 狀態指標",
        "wow_agents": "已載入代理數",
        "wow_workitems": "工作項目數",
        "wow_risks": "風險項目數",
        "wow_ready": "可開始協調",
        "chat_prompt": "優化提示（針對目前結果進一步要求）",
        "run_refinement": "執行優化",
        "apply_refinement": "套用優化片段至計畫",
        "nodes_label": "節點為工作項目，箭頭為相依關係。",
    },
}


# -------------------------------------------
# Session State Initialization & Theme / Lang
# -------------------------------------------

def init_session_state():
    if "theme" not in st.session_state:
        st.session_state["theme"] = "dark"
    if "lang" not in st.session_state:
        st.session_state["lang"] = LANG_EN
    if "painter_style" not in st.session_state:
        st.session_state["painter_style"] = PAINTER_STYLES[0]
    if "api_keys" not in st.session_state:
        st.session_state["api_keys"] = {
            "openai": None,
            "gemini": None,
            "anthropic": None,
            "grok": None,
        }
    if "agents_config" not in st.session_state:
        st.session_state["agents_config"] = {"agents": []}
    if "skills" not in st.session_state:
        st.session_state["skills"] = {}
    if "project_plan" not in st.session_state:
        st.session_state["project_plan"] = None
    if "last_agent_output" not in st.session_state:
        st.session_state["last_agent_output"] = ""
    if "handoff_buffer" not in st.session_state:
        st.session_state["handoff_buffer"] = ""
    if "refined_fragment" not in st.session_state:
        st.session_state["refined_fragment"] = ""
    if "orchestrator_settings" not in st.session_state:
        st.session_state["orchestrator_settings"] = {
            "model": SUPPORTED_MODELS[0],
            "max_tokens": DEFAULT_MAX_TOKENS,
            "system_prompt": "",
        }


def get_ui_text() -> Dict[str, str]:
    return UI_TEXT.get(st.session_state["lang"], UI_TEXT[LANG_EN])


def apply_custom_theme():
    style_name = st.session_state["painter_style"]
    bg, accent = PAINTER_STYLE_PALETTES.get(
        style_name,
        ("linear-gradient(135deg,#020617,#1f2937)", "#3b82f6"),
    )
    is_dark = st.session_state["theme"] == "dark"

    text_color = "#e5e7eb" if is_dark else "#111827"
    card_bg = "rgba(15,23,42,0.85)" if is_dark else "rgba(255,255,255,0.9)"
    border_color = "rgba(148,163,184,0.4)" if is_dark else "rgba(148,163,184,0.6)"

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {bg} !important;
            color: {text_color} !important;
        }}
        .wow-card {{
            background: {card_bg};
            border-radius: 16px;
            padding: 1.1rem 1.25rem;
            border: 1px solid {border_color};
            box-shadow: 0 18px 45px rgba(15,23,42,0.6);
            backdrop-filter: blur(16px);
        }}
        .wow-title {{
            font-size: 1.8rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            color: {accent};
        }}
        .wow-pill {{
            display: inline-flex;
            align-items: center;
            padding: 0.25rem 0.7rem;
            border-radius: 999px;
            border: 1px solid {border_color};
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: {text_color};
        }}
        .wow-label {{
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            opacity: 0.7;
        }}
        .wow-value {{
            font-size: 1.3rem;
            font-weight: 600;
        }}
        .wow-accent {{
            color: {accent};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# -------------------------
# Config Loading & Parsing
# -------------------------

def load_agents_config() -> Dict[str, Any]:
    path = "agents.yaml"
    if not os.path.exists(path):
        return {"agents": []}
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if "agents" not in cfg:
        cfg["agents"] = []
    return cfg


def parse_skills_md() -> Dict[str, Dict[str, Any]]:
    path = "SKILL.md"
    if not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    skills: Dict[str, Dict[str, Any]] = {}
    blocks = re.split(r"^#\s*Skill:\s*", content, flags=re.MULTILINE)
    for block in blocks[1:]:
        lines = block.splitlines()
        if not lines:
            continue
        first_line = lines[0].strip()
        skill_id = first_line
        rest = "\n".join(lines[1:])
        desc_match = re.search(r"\*\*Description:\*\*\s*(.*)", rest)
        params_match = re.search(r"\*\*Parameters:\*\*\s*(.*)", rest)
        skills[skill_id] = {
            "id": skill_id,
            "description": desc_match.group(1).strip() if desc_match else "",
            "parameters": params_match.group(1).strip() if params_match else "",
            "raw": rest.strip(),
        }
    return skills


def refresh_config():
    st.session_state["agents_config"] = load_agents_config()
    st.session_state["skills"] = parse_skills_md()


# ----------------------------
# API Keys & LLM Client Helper
# ----------------------------

def init_api_keys_from_env():
    keys = st.session_state["api_keys"]
    if keys["openai"] is None:
        env_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY")
        if env_key:
            keys["openai"] = env_key
    if keys["gemini"] is None:
        env_key = os.getenv("GEMINI_API_KEY")
        if env_key:
            keys["gemini"] = env_key
    if keys["anthropic"] is None:
        env_key = os.getenv("ANTHROPIC_API_KEY")
        if env_key:
            keys["anthropic"] = env_key
    if keys["grok"] is None:
        env_key = os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY")
        if env_key:
            keys["grok"] = env_key
    st.session_state["api_keys"] = keys


def detect_provider(model_name: str) -> str:
    mn = model_name.lower()
    if mn.startswith("gpt-"):
        return "openai"
    if mn.startswith("gemini-"):
        return "gemini"
    if mn.startswith("claude-") or "sonnet" in mn or "haiku" in mn or "anthropic" in mn:
        return "anthropic"
    if mn.startswith("grok-"):
        return "grok"
    return "openai"


def call_llm(
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    provider = detect_provider(model)
    keys = st.session_state["api_keys"]

    if provider == "openai":
        api_key = keys.get("openai")
        if not api_key:
            raise RuntimeError("Missing OpenAI API key.")
        client = openai.OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt or ""},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content

    elif provider == "gemini":
        api_key = keys.get("gemini")
        if not api_key:
            raise RuntimeError("Missing Gemini API key.")
        genai.configure(api_key=api_key)
        prompt = system_prompt + "\n\n" + user_prompt if system_prompt else user_prompt
        model_obj = genai.GenerativeModel(model)
        resp = model_obj.generate_content(prompt)
        return resp.text

    elif provider == "anthropic":
        api_key = keys.get("anthropic")
        if not api_key:
            raise RuntimeError("Missing Anthropic API key.")
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt or "",
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = ""
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                text += block.text
        return text

    elif provider == "grok":
        api_key = keys.get("grok")
        if not api_key:
            raise RuntimeError("Missing Grok API key.")
        grok_client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
        )
        resp = grok_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt or ""},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content

    else:
        raise RuntimeError(f"Unknown provider for model {model}")


# ---------------------------------
# Orchestrator Prompt & JSON Helper
# ---------------------------------

def build_orchestrator_system_prompt() -> str:
    agents = st.session_state["agents_config"]["agents"]
    skills = st.session_state["skills"]

    skill_summaries = []
    for sid, s in skills.items():
        skill_summaries.append(
            f"- {sid}: {s.get('description','')} (params: {s.get('parameters','')})"
        )
    skills_block = "\n".join(skill_summaries) if skill_summaries else "None."

    agent_summaries = []
    for a in agents:
        agent_summaries.append(
            f"- id: {a.get('id')} | name: {a.get('name')} | role: {a.get('role')} | "
            f"capabilities: {', '.join(a.get('capabilities', []))}"
        )
    agents_block = "\n".join(agent_summaries) if agent_summaries else "None."

    return f"""
You are the Orchestrator for an Agentic AI Project Planning system.

You must read an unstructured project or tender description, then output a JSON object
strictly matching the following TypeScript interface (no extra keys):

interface ProjectPlan {{
  meta: {{
    title: string;
    summary: string;
    domain: string;
  }};
  workItems: Array<{{ 
    id: string;
    title: string;
    description: string;
    assignedAgentId: string;
    complexity: "Low" | "Medium" | "High";
    phase: string;
  }}>;
  timeline: Array<{{
    phaseName: string;
    startDate: string;
    duration: string;
    milestones: string[];
  }}>;
  risks: Array<{{
    description: string;
    impact: number;
    probability: number;
    mitigationStrategy: string;
  }}>;
  dependencies: Array<{{
    source: string;
    target: string;
    type: "Blocking" | "Informational";
  }}>;
}}

Agents available (from agents.yaml):
{agents_block}

Skills available (from SKILL.md):
{skills_block}

Rules:
- Use assignedAgentId values that match existing agent ids when possible.
- Ensure IDs like workItems[i].id are unique strings (e.g., "1.1", "2.3").
- Return ONLY valid JSON. Do NOT wrap JSON in markdown fences. Do not add comments.
"""


def parse_json_from_llm(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except Exception as e:
            raise ValueError(f"Failed to parse model output as JSON. Error: {e}")
    raise ValueError("No JSON object found in model output.")


# ---------------------------
# Visualization – WOW & Views
# ---------------------------

def render_wow_status():
    ui = get_ui_text()
    plan = st.session_state["project_plan"]
    agents = st.session_state["agents_config"]["agents"]
    skills = st.session_state["skills"]

    work_items_count = len(plan.get("workItems", [])) if plan else 0
    risks_count = len(plan.get("risks", [])) if plan else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="wow-card">
              <div class="wow-label">{ui["wow_agents"]}</div>
              <div class="wow-value wow-accent">{len(agents)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="wow-card">
              <div class="wow-label">Skills</div>
              <div class="wow-value wow-accent">{len(skills)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="wow-card">
              <div class="wow-label">{ui["wow_workitems"]}</div>
              <div class="wow-value wow-accent">{work_items_count}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
            <div class="wow-card">
              <div class="wow-label">{ui["wow_risks"]}</div>
              <div class="wow-value wow-accent">{risks_count}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_work_breakdown(plan: Dict[str, Any]):
    ui = get_ui_text()
    st.subheader(ui["work_breakdown"])
    work_items = plan.get("workItems", [])
    if not work_items:
        st.info("No work items in the plan yet.")
        return

    rows = []
    for wi in work_items:
        rows.append(
            {
                "ID": wi.get("id"),
                "Title": wi.get("title"),
                "Description": wi.get("description"),
                "Agent": wi.get("assignedAgentId"),
                "Complexity": wi.get("complexity"),
                "Phase": wi.get("phase"),
            }
        )
    st.dataframe(rows, hide_index=True, use_container_width=True)


def render_timeline(plan: Dict[str, Any]):
    ui = get_ui_text()
    st.subheader(ui["timeline"])
    timeline = plan.get("timeline", [])
    if not timeline:
        st.info("No timeline information available.")
        return
    for phase in timeline:
        with st.expander(f"{phase.get('phaseName','(Phase)')} – {phase.get('duration','')}"):
            st.write(f"Start: {phase.get('startDate','')}")
            mstones = phase.get("milestones", [])
            if mstones:
                st.markdown("**Milestones:**")
                for m in mstones:
                    st.markdown(f"- {m}")


def render_agent_matrix(plan: Dict[str, Any]):
    ui = get_ui_text()
    st.subheader(ui["agent_matrix"])
    work_items = plan.get("workItems", [])
    agents_by_id = {
        a.get("id"): a for a in st.session_state["agents_config"]["agents"]
    }

    if not work_items:
        st.info("No work items in the plan.")
        return

    rows = []
    for wi in work_items:
        agent_id = wi.get("assignedAgentId")
        agent = agents_by_id.get(agent_id)
        rows.append(
            {
                "Work Item ID": wi.get("id"),
                "Title": wi.get("title"),
                "Agent ID": agent_id,
                "Agent Name": agent.get("name") if agent else "",
                "Agent Role": agent.get("role") if agent else "",
            }
        )
    st.dataframe(rows, hide_index=True, use_container_width=True)


def render_risk_heatmap(plan: Dict[str, Any]):
    ui = get_ui_text()
    st.subheader(ui["risk_heatmap"])
    risks = plan.get("risks", [])
    if not risks:
        st.info("No risks defined in the plan.")
        return

    cols = st.columns(3)
    with cols[0]:
        st.markdown("**High Impact / High Probability**")
        for r in risks:
            if r.get("impact", 0) >= 7 and r.get("probability", 0) >= 7:
                st.markdown(f"- {r.get('description')}")
    with cols[1]:
        st.markdown("**High Impact / Medium Probability**")
        for r in risks:
            if r.get("impact", 0) >= 7 and 4 <= r.get("probability", 0) < 7:
                st.markdown(f"- {r.get('description')}")
    with cols[2]:
        st.markdown("**Medium Impact / Low Probability**")
        for r in risks:
            if 4 <= r.get("impact", 0) < 7 and r.get("probability", 0) < 4:
                st.markdown(f"- {r.get('description')}")


def render_dependency_graph(plan: Dict[str, Any]):
    ui = get_ui_text()
    st.subheader(ui["dependencies"])
    deps = plan.get("dependencies", [])
    if not deps:
        st.info("No dependencies defined.")
        return

    work_items = {w.get("id"): w for w in plan.get("workItems", [])}
    nodes = []
    edges = []
    for wid, w in work_items.items():
        title = w.get("title", "").replace('"', "'")
        nodes.append(f'"{wid}" [label="{wid}: {title}"];')
    for d in deps:
        s = d.get("source")
        t = d.get("target")
        dep_type = d.get("type", "Informational")
        if not s or not t:
            continue
        color = "red" if dep_type == "Blocking" else "gray"
        edges.append(f'"{s}" -> "{t}" [color="{color}"];')

    dot = "digraph G {\n" + "\n".join(nodes) + "\n" + "\n".join(edges) + "\n}"
    st.graphviz_chart(dot)
    st.caption(ui["nodes_label"])


# --------------------------
# Agent Execution / Chaining
# --------------------------

def render_agents_tab():
    ui = get_ui_text()
    plan = st.session_state["project_plan"]
    agents = st.session_state["agents_config"]["agents"]
    if not agents:
        st.info("No agents loaded from agents.yaml.")
        return

    st.subheader(ui["agents_tab"])

    st.markdown(f"**{ui['shared_handoff']}**")
    # Use a separate widget key and sync to internal state
    st.text_area(
        label="",
        value=st.session_state["handoff_buffer"],
        key="handoff_buffer_widget",
        height=120,
    )
    st.session_state["handoff_buffer"] = st.session_state.get(
        "handoff_buffer_widget", ""
    )

    st.markdown("---")

    if not plan:
        st.info(ui["plan_missing"])
        return

    work_items = plan.get("workItems", [])
    if not work_items:
        st.info("No work items to assign agents to.")
        return

    agent_to_items: Dict[str, List[Dict[str, Any]]] = {a["id"]: [] for a in agents}
    for wi in work_items:
        aid = wi.get("assignedAgentId")
        if aid in agent_to_items:
            agent_to_items[aid].append(wi)

    for agent in agents:
        aid = agent.get("id")
        with st.expander(f"Agent: {agent.get('name')} ({aid}) – {agent.get('role')}"):
            items = agent_to_items.get(aid, [])
            if not items:
                st.caption("No work items currently assigned.")
            else:
                for wi in items:
                    st.markdown(f"**[{wi.get('id')}] {wi.get('title')}**")
                    st.caption(wi.get("description", ""))

                    default_input = (
                        f"You are {agent.get('name')} with role {agent.get('role')}.\n"
                        f"Task: {wi.get('title')} (ID: {wi.get('id')})\n"
                        f"Description: {wi.get('description')}\n\n"
                        f"Handoff context (if any):\n{st.session_state.get('handoff_buffer','')}\n"
                    )
                    user_input_key = f"agent_input_{aid}_{wi.get('id')}"
                    agent_input = st.text_area(
                        ui["agent_input"],
                        value=default_input,
                        key=user_input_key,
                        height=160,
                    )

                    col_run, col_view = st.columns([1, 1])
                    with col_run:
                        btn_key = f"run_{aid}_{wi.get('id')}"
                        if st.button(ui["run_agent"], key=btn_key):
                            try:
                                model = agent.get("model") or st.session_state[
                                    "orchestrator_settings"
                                ]["model"]
                                max_tokens = st.session_state["orchestrator_settings"][
                                    "max_tokens"
                                ]
                                system_prompt = agent.get("system_prompt", "")
                                result = call_llm(
                                    model=model,
                                    system_prompt=system_prompt,
                                    user_prompt=agent_input,
                                    max_tokens=max_tokens,
                                )
                                st.session_state["last_agent_output"] = result
                                # Update internal state (not widget key)
                                st.session_state["handoff_buffer"] = result
                                st.session_state["handoff_buffer_widget"] = result
                                st.success("Agent execution completed.")
                            except Exception as e:
                                st.error(f"Agent call failed: {e}")

                    with col_view:
                        view_mode = st.radio(
                            ui["agent_result_view"],
                            options=[ui["text_view"], ui["markdown_view"]],
                            key=f"view_{aid}_{wi.get('id')}",
                            horizontal=True,
                        )

                    if st.session_state["last_agent_output"]:
                        if view_mode == ui["markdown_view"]:
                            st.markdown(st.session_state["last_agent_output"])
                        else:
                            st.text(st.session_state["last_agent_output"])


# --------------------------
# Refinement / Prompt on Plan
# --------------------------

def render_refinement_tab():
    ui = get_ui_text()
    plan = st.session_state["project_plan"]
    st.subheader(ui["chat_tab"])
    if not plan:
        st.info(ui["plan_missing"])
        return

    refinement_prompt = st.text_area(
        ui["chat_prompt"],
        value="For Item 10, strictly focus on Generative AI for label recognition.",
        height=140,
    )
    if st.button(ui["run_refinement"]):
        try:
            model = st.session_state["orchestrator_settings"]["model"]
            max_tokens = 2000
            system_prompt = (
                "You are a JSON patch generator for the current project plan. "
                "You MUST return only a valid JSON fragment that can be merged into "
                "the existing plan (e.g., an updated workItems array or a single updated object)."
            )
            user_prompt = (
                "Current plan JSON:\n"
                + json.dumps(plan, ensure_ascii=False, indent=2)
                + "\n\nUser refinement request:\n"
                + refinement_prompt
                + "\n\nReturn ONLY the updated JSON fragment, no comments, no markdown."
            )
            result = call_llm(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
            )
            st.session_state["refined_fragment"] = result
            st.success("Refinement result generated (raw JSON fragment below).")
        except Exception as e:
            st.error(f"Refinement LLM call failed: {e}")

    if st.session_state["refined_fragment"]:
        st.markdown("**Refined JSON fragment (editable before apply):**")
        frag = st.text_area(
            "",
            value=st.session_state["refined_fragment"],
            key="refined_fragment",
            height=200,
        )
        if st.button(ui["apply_refinement"]):
            try:
                fragment_obj = parse_json_from_llm(frag)
                plan = st.session_state["project_plan"] or {}
                for key in ["meta", "workItems", "timeline", "risks", "dependencies"]:
                    if key in fragment_obj:
                        plan[key] = fragment_obj[key]
                st.session_state["project_plan"] = plan
                st.success("Refined fragment applied to current plan.")
            except Exception as e:
                st.error(f"Failed to apply fragment: {e}")


# -------------------------
# Orchestrator Runner (UI)
# -------------------------

def run_orchestrator_ui():
    ui = get_ui_text()
    st.subheader(ui["orchestrator_settings"])

    settings = st.session_state["orchestrator_settings"]
    col1, col2 = st.columns([2, 1])
    with col1:
        model = st.selectbox(
            ui["model"],
            options=SUPPORTED_MODELS,
            index=SUPPORTED_MODELS.index(settings["model"])
            if settings["model"] in SUPPORTED_MODELS
            else 0,
        )
    with col2:
        max_tokens = st.number_input(
            ui["max_tokens"],
            min_value=256,
            max_value=64000,
            value=settings.get("max_tokens", DEFAULT_MAX_TOKENS),
            step=512,
        )

    system_prompt_override = st.text_area(
        ui["system_prompt"],
        value=settings.get("system_prompt", ""),
        height=150,
    )

    settings["model"] = model
    settings["max_tokens"] = max_tokens
    settings["system_prompt"] = system_prompt_override
    st.session_state["orchestrator_settings"] = settings

    project_text = st.text_area(
        ui["project_input"],
        height=280,
        key="project_input",
        value="",
    )

    if st.button(ui["run_orchestrator"]):
        if not project_text.strip():
            st.warning("Please paste a project / tender description first.")
        else:
            try:
                base_system_prompt = (
                    system_prompt_override
                    if system_prompt_override.strip()
                    else build_orchestrator_system_prompt()
                )
                user_prompt = (
                    "Project / tender description:\n\n"
                    + project_text
                    + "\n\nNow generate the ProjectPlan JSON as specified."
                )
                with st.spinner("Orchestrating project plan with selected model..."):
                    raw = call_llm(
                        model=model,
                        system_prompt=base_system_prompt,
                        user_prompt=user_prompt,
                        max_tokens=max_tokens,
                    )
                plan = parse_json_from_llm(raw)
                st.session_state["project_plan"] = plan
                st.success("Project plan generated successfully.")
            except Exception as e:
                st.error(f"Orchestrator call failed: {e}")


# ---------------------
# API Key Config (UI)
# ---------------------

def render_api_key_section():
    ui = get_ui_text()
    st.subheader(ui["api_section"])
    st.caption(ui["api_hint"])

    keys = st.session_state["api_keys"]

    def masked_placeholder(value: Optional[str]) -> str:
        if value:
            return "******** (from env / stored)"
        return ""

    openai_input = st.text_input(
        ui["openai_key"],
        type="password",
        placeholder=masked_placeholder(keys.get("openai")),
    )
    gemini_input = st.text_input(
        ui["gemini_key"],
        type="password",
        placeholder=masked_placeholder(keys.get("gemini")),
    )
    anthropic_input = st.text_input(
        ui["anthropic_key"],
        type="password",
        placeholder=masked_placeholder(keys.get("anthropic")),
    )
    grok_input = st.text_input(
        ui["grok_key"],
        type="password",
        placeholder=masked_placeholder(keys.get("grok")),
    )

    if st.button(get_ui_text()["save_keys"]):
        if openai_input.strip():
            keys["openai"] = openai_input.strip()
        if gemini_input.strip():
            keys["gemini"] = gemini_input.strip()
        if anthropic_input.strip():
            keys["anthropic"] = anthropic_input.strip()
        if grok_input.strip():
            keys["grok"] = grok_input.strip()
        st.session_state["api_keys"] = keys
        st.success("API keys updated for this session.")


# ----------------------
# Layout & Main Entrypoint
# ----------------------

def sidebar_controls():
    ui = get_ui_text()
    st.sidebar.markdown("### UI")

    # Use widget keys different from internal state keys to avoid conflicts
    theme = st.sidebar.radio(
        ui["theme"],
        options=["light", "dark"],
        key="theme_widget",
        horizontal=True,
        index=0 if st.session_state["theme"] == "light" else 1,
    )
    st.session_state["theme"] = theme

    lang = st.sidebar.radio(
        ui["language"],
        options=[LANG_EN, LANG_ZH],
        key="lang_widget",
        horizontal=True,
        index=0 if st.session_state["lang"] == LANG_EN else 1,
    )
    st.session_state["lang"] = lang

    apply_custom_theme()
    ui = get_ui_text()

    st.sidebar.markdown("### 🎨 Style")
    st.sidebar.selectbox(
        ui["painter_style"],
        options=PAINTER_STYLES,
        index=PAINTER_STYLES.index(st.session_state["painter_style"])
        if st.session_state["painter_style"] in PAINTER_STYLES
        else 0,
        key="painter_style",
    )

    if st.sidebar.button(ui["jackpot"]):
        st.session_state["painter_style"] = random.choice(PAINTER_STYLES)

    st.sidebar.markdown("---")
    if st.sidebar.button(get_ui_text()["refresh_config"]):
        refresh_config()
        st.sidebar.success("Config reloaded.")


def main():
    st.set_page_config(
        page_title="Agentic AI Project Orchestrator",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_session_state()
    init_api_keys_from_env()
    apply_custom_theme()
    refresh_config()

    sidebar_controls()
    ui = get_ui_text()

    st.markdown(
        f"""
        <div class="wow-card" style="margin-bottom:1.2rem;">
          <div class="wow-pill">Agentic AI • Multi-LLM • Visualization</div>
          <div class="wow-title" style="margin-top:0.4rem;">{ui["title"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_wow_status()

    tabs = st.tabs(
        [
            ui["dashboard"],
            ui["agents_tab"],
            ui["chat_tab"],
            ui["skills_tab"],
            ui["config"],
        ]
    )

    with tabs[0]:
        run_orchestrator_ui()
        plan = st.session_state["project_plan"]
        if plan:
            st.markdown("---")
            render_work_breakdown(plan)
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                render_timeline(plan)
            with col2:
                render_agent_matrix(plan)
            st.markdown("---")
            col3, col4 = st.columns(2)
            with col3:
                render_risk_heatmap(plan)
            with col4:
                render_dependency_graph(plan)

    with tabs[1]:
        render_agents_tab()

    with tabs[2]:
        render_refinement_tab()

    with tabs[3]:
        st.subheader(ui["skills_tab"])
        skills = st.session_state["skills"]
        if not skills:
            st.info("No skills parsed from SKILL.md.")
        else:
            for sid, s in skills.items():
                with st.expander(f"{sid}"):
                    st.markdown(f"**Description:** {s.get('description','')}")
                    if s.get("parameters"):
                        st.markdown(f"**Parameters:** {s.get('parameters','')}")
                    if s.get("raw"):
                        st.code(s["raw"], language="markdown")

    with tabs[4]:
        render_api_key_section()
        st.markdown("---")
        st.markdown("**Raw agents.yaml preview:**")
        st.json(st.session_state["agents_config"], expanded=False)


if __name__ == "__main__":
    main()
