# Kavriq Agentic AI v2: Publication Plan

## Part 1: Foundations

### 1. What Is an Agentic System?

- LLM call vs chatbot vs workflow vs agent
- Perception, state, decision, action, and feedback
- Goal-directed behavior
- Autonomy as a spectrum
- Reactive systems and environmental uncertainty
- When not to use an agent

### 2. Anatomy of an Agent

- Environment
- Observation
- Internal state
- Decision mechanism
- Action
- Feedback
- Termination condition
- Human and external-system boundaries
- Minimal deterministic agent loop without an LLM

### 3. The Engineering of Uncertainty

- Why agent demos look capable but fail in real workflows
- Stochastic model behavior vs deterministic system expectations
- Uncertainty, time, and state as core engineering concerns
- Reasoning, action, execution, and control as the four pillars

## Part 2: The Model Inside the Agent

### 4. What an LLM Contributes

- Tokenization and token sequences
- Embeddings and contextual representations
- Transformer intuition
- Context windows
- Next-token prediction
- Fluency vs correctness
- LLMs as probabilistic decision components

### 5. Sampling and Behavior

- Logits and softmax
- Temperature
- Top-p sampling
- Determinism and reproducibility
- Best-of-N sampling
- Quality, diversity, latency, and cost trade-offs

### 6. Instructions and Structured Decisions

- System and user instructions
- Zero-shot and few-shot prompting
- Structured output
- Schema validation
- Classification and routing
- Separating instructions from untrusted data
- Typed routing component

## Part 3: From Model Calls to Agent Loops

### 7. The Agent Loop

- Observe-reason-act cycle
- Explicit state transitions
- Iteration limits
- Stop conditions
- Error paths
- Deterministic workflows vs open-ended loops
- Minimal agent loop in plain Python

### 8. Tools and Function Calling

- Tool definitions and schemas
- Tool selection
- Argument generation and validation
- Tool execution and observation
- Tool errors, retries, and timeouts
- Side effects and idempotency
- Permission boundaries
- Approval before consequential actions

### 9. Routing and Workflow Patterns

- Intent routing
- Sequential workflows
- Parallel execution
- Conditional branches
- Map-reduce workflows
- Supervisor-worker pattern
- Event-driven workflows
- Graphs vs loops

### 10. Why Agents Fail

- Demo vs production gap
- State drift
- Partial execution
- Tool failure
- Infinite retry loops
- Missing validation
- Recovery, escalation, and stop conditions

### 11. State, Checkpoints, and Recovery

- Conversation state vs execution state
- Durable state
- Checkpointing
- Interrupt and resume
- Duplicate execution
- Idempotent actions
- Recovery after partial failure

### 12. From DAGs to State Machines

- Limits of static chains and DAGs
- Loops, branching, retries, and dynamic decisions
- State, transition, event, and action
- Mapping agent execution to state machines
- Cyclic reasoning and fallback paths

## Part 4: Knowledge and Memory

### 13. Context Is Not Memory

- Context-window contents
- Working state
- Semantic memory
- Episodic memory
- Procedural memory
- External application state
- Memory writing, selection, expiry, and retrieval

### 14. Retrieval-Augmented Generation

- Ingestion and chunking
- Embeddings
- Vector search
- Retrieval
- Context construction
- Grounded generation
- Citations and provenance
- Dot product, cosine similarity, top-k retrieval, precision, and recall

### 15. Reliable Retrieval

- Chunk size and overlap
- Metadata filters
- Hybrid search
- Reranking
- Query rewriting
- Context compression
- Retrieval evaluation
- Separating retrieval failure from generation failure

### 16. Agentic RAG

- Fixed RAG vs agent-directed retrieval
- Query planning and decomposition
- Iterative retrieval
- Selecting among knowledge tools
- Evidence sufficiency
- Retrieval stop conditions
- Cost and latency controls

### 17. GraphRAG and Structured Knowledge

- Knowledge graphs and relationships
- Entity and relation extraction
- Traversal and neighborhood retrieval
- Vector search vs graph traversal
- When GraphRAG is justified
- Operational complexity

## Part 5: Planning, Reasoning, and Learning

### 18. What Does It Mean for an Agent to Reason?

- Generated reasoning traces
- Decomposition
- Search and verification
- Outcome vs explanation
- Unfaithful rationales
- Private reasoning vs observable evidence
- Verifiable intermediate work

### 19. ReAct

- Interleaving reasoning, actions, and observations
- Tool-use trajectories
- Environment feedback
- Failure modes
- Loop termination
- ReAct-style agent from scratch

### 20. Planning Patterns

- Classical planning intuition
- Plan-and-execute
- Plan-and-Solve
- Dynamic replanning
- Hierarchical decomposition
- Dependency graphs
- Planning horizons
- When planning hurts performance

### 21. Reflection and Self-Correction

- Critique
- Reflection memory
- Evaluator-optimizer loops
- External feedback
- Trajectory revision
- Why self-review alone is unreliable
- Verifiers and deterministic checks

### 22. Search and Inference-Time Compute

- Best-of-N
- Majority voting
- Candidate ranking
- Judge models
- Tree-style exploration
- Compute-quality trade-offs
- Correlated errors

### 23. Learning from Feedback

- Demonstration selection
- Dataset preparation
- Prompt optimization
- DSPy
- Reward signals
- Reinforcement-learning intuition
- Online vs offline improvement
- Lifelong learning risks

## Part 6: Multi-Agent Systems

### 24. Why Multiple Agents?

- Specialization
- Context isolation
- Parallelism
- Different models and permissions
- Added communication, cost, and failure modes
- Single-agent baseline before multi-agent design

### 25. Communication and Coordination

- Message passing
- Shared state
- Handoffs
- Blackboard pattern
- Supervisor pattern
- Peer-to-peer collaboration
- Task allocation
- Termination, starvation, and deadlock

### 26. Voting, Debate, and Consensus

- Independent candidates
- Majority voting
- Debate
- Judge agents
- Game-theoretic intuition
- Shared-model correlation
- Why agreement is not proof of correctness

### 27. Multi-Agent Reliability

- Conflicting goals
- Infinite delegation
- Duplicate work
- Communication overhead
- Permission propagation
- Cost amplification
- Traceability
- Strong single-agent baseline comparisons

## Part 7: Protocols and Frameworks

### 28. Model Context Protocol

- Integration problem MCP solves
- Hosts, clients, and servers
- Tools, resources, and prompts
- Capability discovery
- MCP interaction lifecycle
- Authentication and trust boundaries
- MCP vs ordinary APIs
- Why MCP is not agent memory

### 29. Building an Agent Without a Framework

- Model API
- Typed schemas
- Tool registry
- Explicit state
- Execution loop
- Checkpoints
- Tracing
- Tests

### 30. LangGraph

- Nodes and edges
- Shared state
- Conditional routing
- Checkpoints
- Interrupts
- Durable execution
- Benefits and abstraction costs
- Rebuilding the reference agent in LangGraph

### 31. CrewAI and Role-Based Orchestration

- Roles, tasks, and crews
- Delegation
- Appropriate use cases
- Hidden orchestration complexity
- Comparison with graph-based workflows

### 32. Choosing an Agent Architecture

- Plain Python
- State-machine workflows
- LangGraph
- CrewAI
- Single-agent systems
- Multi-agent systems
- Trade-offs: transparency, durability, testability, recovery, cost, and operational complexity

## Part 8: Evaluation, Safety, and Production

### 33. What Should We Evaluate?

- Final-answer quality
- Task completion
- Tool-selection accuracy
- Tool-argument correctness
- Retrieval quality
- Trajectory efficiency
- Safety compliance
- Cost
- Latency
- Recovery rate

### 34. Evaluating Agent Trajectories

- Outcome-based evaluation
- Step-level evaluation
- Golden trajectories
- Acceptable alternative paths
- Deterministic checks
- LLM-as-judge
- Human evaluation
- Regression datasets
- Offline and online evaluation

### 35. Controlled Agency

- Risks of unbounded agents
- Tool contracts
- Bounded autonomy
- Scoped permissions
- Action gating
- Approval checkpoints
- Separating reasoning, execution, and control

### 36. Human-in-the-Loop Design

- Approval before consequential actions
- Review and correction
- Escalation
- Interrupt and resume
- Confidence and risk thresholds
- Trust and common ground
- Useful control vs ceremonial approval

### 37. Agent Security

- Direct and indirect prompt injection
- Tool poisoning
- Data exfiltration
- Excessive agency
- Least privilege
- Sandboxing
- Secrets management
- Access control
- Audit logs

### 38. Guardrails and Alignment

- Input and output controls
- Policy enforcement
- Tool-level controls
- Architectural guardrails
- Agentic risk
- Alignment limitations
- Why prompt-only controls are insufficient

### 39. Observability

- Traces and spans
- Model calls
- Tool calls
- State transitions
- Token and cost accounting
- Latency
- Failure classification
- User feedback

### 40. Reliability Engineering for Agents

- Timeouts
- Retries and exponential backoff
- Circuit breakers
- Idempotency
- Rate limits
- Token and monetary budgets
- Dead-letter handling
- Compensating actions
- Partial failure
- Graceful degradation
- Model fallback
- Deterministic escape hatches

### 41. Deployment and Operations

- FastAPI service boundary
- Asynchronous execution
- Background workers
- Streaming
- Persistent state
- Authentication
- Docker
- Scaling
- Configuration and secrets
- Production-readiness checklist

## Part 9: Capstone and Synthesis

### 42. Building a Research and Analysis Agent

- Question decomposition
- Multi-source search
- Private knowledge retrieval
- Evidence and provenance tracking
- Safe tool use
- Structured cited analysis
- Human approval
- Pause and resume
- Observable traces
- Regression dataset
- Deployable API

### 43. Breaking the Agent

- Prompt injection
- Malformed tool results
- Unavailable APIs
- Conflicting evidence
- Repeated tool calls
- Runaway iteration
- Context overflow
- Delayed responses
- Partial state loss
- Model substitution

### 44. Production Readiness Review

- Correctness
- Security
- Reliability
- Observability
- Human control
- Latency
- Cost
- Maintainability
- Deployment decision framework
