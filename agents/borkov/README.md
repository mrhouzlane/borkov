Borkov Protocol Optimization Agent
A specialized Coral agent for analyzing and optimizing lending protocol parameters using Markov-process borrower simulations.
The Borkov agent helps protocol designers maximize ROI, minimize default risk, and recommend optimal settings for lending economics (interest rates, transition probabilities, penalties, and more).

Features
Markov-Process Simulation: Model borrower journeys across protocol states (Reset, Building, Established).

Cohort Analysis: Run cohort simulations by archetype to summarize risk and growth properties.

Steady-State Analytics: Compute long-term borrower state distributions for protocol stability.

State Transition Matrix Generation: Outputs Markov matrices for protocol state logic.

Protocol Parameter Optimization: Sweeps key protocol settings (interest rates, transition probabilities, penalties) and recommends design choices to maximize ROI or stability.

Structured, Actionable Insights: Returns clear parameter recommendations, tables, and rationales for every analysis.

Coral Integration: Full integration with the Coral framework for seamless agent collaboration.

Tools Available
borkov_single_journey
Simulate a single borrower's journey under specific protocol parameters.

n_loans: Number of loan cycles (default: 50)

p_repay: Probability of repayment per cycle (default: 0.92)

d: Multiplier increment per successful repayment (default: 0.15)

reset_on_default: Hard reset to base multiplier on default (default: True)

Returns: Dict with journey, default count, final multiplier, and stats

borkov_cohort_analysis
Simulate multiple archetype cohorts and summarize multipliers, defaults, and streaks by borrower group.

n_loans: Loan cycles per borrower (default: 50)

archetypes: Dict of cohort definitions (optional)

Returns: Summary stats grouped by archetype

borkov_transition_matrix
Output the Markov state transition (protocol) matrix for a given repayment probability (p_repay).

p_repay: Repayment success probability

Returns: List of matrix rows/columns

borkov_steady_state
Compute steady-state probabilities for each protocol borrower state.

p_repay: Repayment probability

Returns: Dict of population share per borrower state (Reset, Building, Established)

borkov_optimize_roi
Run parameter sweeps over interest rates and transition probabilities to find the design that maximizes ROI and protocol health.

interest_rate_start, interest_rate_end, interest_rate_step: Grid for interest rate search

p_repay_start, p_repay_end, p_repay_step: Grid for p_repay search

n_loans: Loan cycles per sim (default: 50)

penalty: Default penalty factor (default 0.5)

Returns: Best settings, projected ROI, steady-state, and top results table

Configuration
Required Environment Variables
GOOGLE_API_KEY: Google API key for Gemini AI model

CORAL_SSE_URL: Coral server URL for SSE connection

CORAL_AGENT_ID: Unique agent identifier

Optional (with defaults)
MODEL_NAME: Model to use (default: "gemini-2.0-flash")

MODEL_PROVIDER: Model provider (default: "google_genai")

MODEL_MAX_TOKENS: Max tokens (default: "16000")

MODEL_TEMPERATURE: Model temperature (default: "0.3")

TIMEOUT_MS: Connection timeout in ms (default: "300")

Usage
Running the Agent
bash
./run_agent.sh borkov_main.py

# Or directly:
uv run borkov_main.py
Docker Example
bash
docker build -t borkov-agent .
docker run -e GOOGLE_API_KEY=your_key -e CORAL_SSE_URL=your_url borkov-agent
Agent Behavior
The agent continuously:

Listens for protocol design/analysis tasks via Coral's event system.

Parses and simulates appropriate scenarios (single, cohort, matrix, optimization grid, etc).

Recommends parameter or design changes for maximum ROI or minimum risk.

Returns structured results (bullet points, tables, and rationale) to the requesting agent or user.

Retries and logs errors to ensure robust operation.

Protocol State Model
Borrowers have three possible Markov states:

Reset/New: Multiplier at protocol base

Building: Accumulating positive history (1.0 < multiplier < 3.0)

Established: Long, successful history (multiplier ≥ 3.0)

Protocol transitions and payoff logic are fully customizable per simulation.

Dependencies
Python (3.10+)

NumPy, pandas

LangChain LLM ecosystem

Coral MCP adapters for agent orchestration

Development
bash
uv sync
uv run python borkov_main.py
Integration with Coral
Borkov is designed as a plug-and-play protocol analytics specialist in the Coral agent ecosystem, so it can receive, execute, and report on any design optimization task.

Example: Design Optimization Query
Input:
"Should I increase the penalty for defaults or lower the transition probability from Building to Established to boost long-term protocol ROI?"

Agent Output:

Simulates grid search over penalty and transition probability.

Returns:

“Best result at penalty = 0.6, transition probability = 0.91.”

“Projected ROI: 14.8%; Default rate decreases by 8%.”

“Recommendation: Increase penalty to 0.6 and transition probability to 0.91 for optimal sustainable growth.”

Borkov: for smarter, simulation-driven lending protocol design.