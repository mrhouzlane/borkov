import os
import json
import asyncio
import traceback
from dotenv import load_dotenv
from typing import Dict, Any, Optional, List
import numpy as np
import urllib.parse


from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import tool

from borkov_simulation import (
    simulate_borkov_journey,
    mass_borkov_simulation,
    create_borkov_matrix,
    steady_state_analysis
)

# ----------------------------- Borkov Agent Tools ---------------------------------

@tool
def borkov_single_journey(
    n_loans: int = 50,
    p_repay: float = 0.92,
    d: float = 0.15,
    reset_on_default: bool = True
) -> Dict[str, Any]:
    """
    Simulate a single borrower's journey under Markov-process lending protocol.
    Returns stats, journey, default count, and final multiplier.
    """
    result = simulate_borkov_journey(n_loans, p_repay, d, reset_on_default)
    return {
        "success": True,
        "params": {
            "n_loans": n_loans, "p_repay": p_repay, "d": d, "reset_on_default": reset_on_default
        },
        **result
    }

@tool
def borkov_cohort_analysis(
    n_loans: int = 50,
    archetypes: Optional[Dict[str, Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Simulate a protocol cohort of multiple archetypes.
    Returns summary table grouped by archetype.
    """
    if archetypes is None:
        archetypes = {
            'Diamond Hands': {'p_repay': 0.9, 'count': 300, 'color': 'green'},
            'Paper Hands': {'p_repay': 0.92, 'count': 500, 'color': 'orange'},
            'Degen Traders': {'p_repay': 0.95, 'count': 200, 'color': 'red'}
        }
    df, _ = mass_borkov_simulation(None, n_loans, archetypes)
    summary = df.groupby('archetype')[['final_multiplier', 'defaults', 'max_streak']].agg(['mean', 'std']).round(3)
    return {
        "success": True,
        "summary_table": summary.to_dict()
    }

@tool
def borkov_transition_matrix(p_repay: float):
    """
    Return the Borkov Markov state transition matrix for protocol analysis.
    """
    matrix = create_borkov_matrix(p_repay)
    return {
        "success": True,
        "p_repay": p_repay,
        "matrix": matrix.tolist()
    }

@tool
def borkov_steady_state(p_repay: float):
    """
    Compute steady state borrower state probabilities for a given Markov repayment parameter.
    """
    matrix = create_borkov_matrix(p_repay)
    steady = steady_state_analysis(matrix)
    return {
        "success": True,
        "p_repay": p_repay,
        "steady_state_probs": {
            "Reset": round(float(steady[0]), 4),
            "Building": round(float(steady[1]), 4),
            "Established": round(float(steady[2]), 4)
        }
    }

@tool
def borkov_optimize_roi(
    interest_rate_start: float = 0.05,
    interest_rate_end: float = 0.18,
    interest_rate_step: float = 0.01,
    p_repay_start: float = 0.80,
    p_repay_end: float = 0.98,
    p_repay_step: float = 0.01,
    n_loans: int = 50,
    penalty: float = 0.5
) -> Dict[str, Any]:
    """
    Run a simulation grid over ranges of interest rates and p_repay (transition success)
    to find the combination that yields highest ROI (final multiplier), plus recommendations.
    All archetypes use the same parameters.
    Returns best params, expected ROI, and optimization sweep.
    """
    best_roi = -np.inf
    best_params = {}
    best_steady = None
    results = []

    for ir in np.arange(interest_rate_start, interest_rate_end + 0.001, interest_rate_step):
        for p_repay in np.arange(p_repay_start, p_repay_end + 0.001, p_repay_step):
            # Simulate a single journey as a proxy for per-borrower ROI
            sim = simulate_borkov_journey(n_loans=n_loans, p_repay=p_repay, d=ir, reset_on_default=True)
            matrix = create_borkov_matrix(p_repay)
            steady = steady_state_analysis(matrix)
            avg_mult = sim["final_multiplier"]
            roi = avg_mult * ir  # Proxy: final state * interest rate
            
            results.append({
                "interest_rate": round(ir, 3),
                "p_repay": round(p_repay, 3),
                "projected_roi": round(roi, 4),
                "steady_state_established": round(float(steady[2]), 4)
            })
            if roi > best_roi:
                best_roi = roi
                best_params = {
                    "interest_rate": round(ir, 3),
                    "p_repay": round(p_repay, 3),
                    "penalty": penalty
                }
                best_steady = [round(float(x), 4) for x in steady]

    recommendation = (
        f"Set interest_rate to {best_params['interest_rate']:.3f}, "
        f"p_repay (success probability) to {best_params['p_repay']:.3f} "
        f"for maximum projected ROI ({best_roi:.4f}). "
        f"At this setting, established borrower steady-state = {best_steady[2]:.4f}."
    )
    return {
        "success": True,
        "best_params": best_params,
        "best_projected_roi": round(best_roi, 4),
        "recommendation": recommendation,
        "best_steady_state_probs": {
            "Reset": best_steady[0],
            "Building": best_steady[1],
            "Established": best_steady[2]
        },
        "top10_results": sorted(results, key=lambda x: -x["projected_roi"])[:10]
    }

# ---------------------- Agent Environment & Startup ---------------------------------

def get_tools_description(tools):
    return "\n".join(
        f"Tool: {tool.name}, Schema: {json.dumps(tool.args).replace('{', '{{').replace('}', '}}')}"
        for tool in tools
    )

async def create_agent(coral_tools, agent_tools):
    coral_tools_description = get_tools_description(coral_tools)
    agent_tools_description = get_tools_description(agent_tools)
    combined_tools = coral_tools + agent_tools

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            f"""You are Borkov, a protocol design advisor specializing in Markov-process lending simulation and optimization.
Your role is to:
- Analyze protocol parameters (interest rates, state transitions, penalties) to maximize ROI and minimize defaults.
- Run Markov-chain cohort simulations, state matrix analysis, and param sweeps on request.
- Recommend concrete parameter or design changes—always explain why.
- When simulating or optimizing, report the best parameter set and provide bullet-pointed recommendations and summary tables.

Available MCP/Coral tools: {coral_tools_description}
Available Borkov analytics/optimization tools: {agent_tools_description}

Always respond using concise technical summary, actionable findings, and supporting stats.
"""
        ),
        ("human", "Start the Borkov protocol design optimization agent."),
        ("placeholder", "{agent_scratchpad}")
    ])

    model = init_chat_model(
        model=os.getenv("MODEL_NAME", "gemini-2.0-flash"),
        model_provider=os.getenv("MODEL_PROVIDER", "google"),
        api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=float(os.getenv("MODEL_TEMPERATURE", "0.3")),
        max_tokens=int(os.getenv("MODEL_MAX_TOKENS", "16000"))
    )
    agent = create_tool_calling_agent(model, combined_tools, prompt)
    return AgentExecutor(agent=agent, tools=combined_tools, verbose=True, handle_parsing_errors=True)

async def main():
    runtime = os.getenv("CORAL_ORCHESTRATION_RUNTIME", None)
    if runtime is None:
        load_dotenv()

    base_url = os.getenv("CORAL_SSE_URL")
    agentID = os.getenv("CORAL_AGENT_ID")

    coral_params = {
        "agentId": agentID,
        "agentDescription": "Borkov: Markov-process lending protocol design and optimization agent."
    }
    query_string = urllib.parse.urlencode(coral_params)
    CORAL_SERVER_URL = f"{base_url}?{query_string}"
    print(f"Connecting to Coral Server: {CORAL_SERVER_URL}")

    timeout = float(os.getenv("TIMEOUT_MS", "300"))
    client = MultiServerMCPClient(
        connections={
            "coral": {
                "transport": "sse",
                "url": CORAL_SERVER_URL,
                "timeout": timeout,
                "sse_read_timeout": timeout,
            }
        }
    )

    print("Multi Server Connection Initialized")
    coral_tools = await client.get_tools(server_name="coral")

    # Register all Borkov analytics and optimization tools
    agent_tools = [
        borkov_single_journey,
        borkov_cohort_analysis,
        borkov_transition_matrix,
        borkov_steady_state,
        borkov_optimize_roi
    ]

    print(f"Coral tools count: {len(coral_tools)}, Agent tools count: {len(agent_tools)}")

    agent_executor = await create_agent(coral_tools, agent_tools)

    while True:
        try:
            print("Starting Borkov agent invocation")
            await agent_executor.ainvoke({"agent_scratchpad": []})
            print("Completed agent invocation, restarting loop")
            await asyncio.sleep(1)
        except Exception as e:
            print(f"Error in agent loop: {e}")
            traceback.print_exc()
            await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(main())
