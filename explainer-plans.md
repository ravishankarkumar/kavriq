# Kavriq Explainers: Publication Plan

## Section

- Name: Explainers
- URL: `/explainers`
- Format: standalone question-led articles

## Planned Clusters

## Cluster 1: How ChatGPT and LLMs Work

### 1. What Happens After You Send a Prompt to ChatGPT?

- Prompt submission
- Tokenization
- Numerical representations
- Context processing
- Next-token probability
- Sampling
- Autoregressive generation
- Decoding generated tokens into readable text

### 2. What Is a Token in AI?

- Tokens vs words
- Subwords, punctuation, and whitespace
- Why models process tokens instead of full words
- How token count affects context, cost, and latency
- Simple tokenization examples

### 3. How Does ChatGPT Choose Its Next Word?

- Next-token prediction
- Logits
- Probability distribution over possible tokens
- Sampling vs always picking the highest-probability token
- Why generation happens one token at a time

### 4. Does ChatGPT Actually Understand Language?

- Pattern recognition vs human understanding
- Contextual representations
- Why fluent answers can still be wrong
- Operational meaning of "understanding"
- Where the analogy breaks down

### 5. Why Does ChatGPT Give Different Answers to the Same Question?

- Sampling randomness
- Temperature
- Top-p sampling
- Prompt wording sensitivity
- Context differences
- Determinism settings

### 6. Why Does ChatGPT Make Things Up?

- Hallucination as unsupported generation
- Lack of guaranteed truth checking
- Missing or weak context
- Confident language vs verified facts
- Retrieval and citation limitations
- When hallucination risk is highest

### 7. What Is a Context Window?

- Context as the text available to the model
- Input tokens and output tokens
- Context limits
- What gets included in a request
- What happens when context becomes too large

### 8. How Does ChatGPT Remember a Conversation?

- Conversation history in context
- Persistent memory vs current chat context
- Why models can appear to forget
- Summarization and compression
- Limits of conversational memory

### 9. Does ChatGPT Search the Internet for Every Answer?

- Model knowledge vs live retrieval
- When browsing/search tools are needed
- Retrieval-augmented answers
- Why non-browsing answers can be outdated
- Difference between generated text and sourced information

### 10. How Can ChatGPT Answer Questions About So Many Subjects?

- Pretraining on broad text
- Statistical patterns across domains
- Generalization
- Limits of memorization
- Why breadth does not guarantee correctness

### 11. What Happens When ChatGPT Runs Out of Context?

- Context overflow
- Truncation
- Loss of earlier details
- Summarization strategies
- Retrieval and memory as workarounds

### 12. Why Can a Smaller AI Model Sometimes Beat a Larger One?

- Task fit
- Training data quality
- Fine-tuning
- Prompting and context quality
- Latency and specialization
- Evaluation by use case rather than parameter count

## Cluster 2: AI Agents

### 13. What Is an AI Agent?

- Agent vs chatbot
- Goal, state, action, and feedback
- Tool use
- Autonomy as a spectrum
- When a system is not really an agent

### 14. How Is an AI Agent Different from a Chatbot?

- Conversational response vs action-taking loop
- State over time
- Tool execution
- Planning and retries
- Consequences and control

### 15. How Can an AI Agent Use Tools?

- Tool definitions
- Tool schemas
- Function calling
- Argument generation
- Tool results as observations
- Validation and permissions

### 16. What Does Function Calling Mean?

- Structured tool requests
- Function name and arguments
- Application-side execution
- Tool result returned to the model
- Difference between calling a function and generating text

### 17. Does an AI Agent Think Before It Acts?

- Reasoning as generated intermediate work
- Planning vs action
- Observable reasoning traces
- Why reasoning text is not always faithful
- Verification before action

### 18. What Is Memory in an AI Agent?

- Current context
- Conversation history
- External memory stores
- Semantic, episodic, and procedural memory
- Retrieval and state persistence

### 19. Why Do AI Agents Get Stuck in Loops?

- Missing stop conditions
- Weak validation
- Repeated tool failures
- Ambiguous goals
- Poor state tracking
- Retry limits and escalation

### 20. Can an AI Agent Take Actions Without Permission?

- Permission scopes
- Consequential actions
- Human approval
- Action gating
- Sandboxing
- Audit logs

### 21. What Is a Multi-Agent System?

- Multiple agents with different roles
- Message passing
- Shared state
- Coordination
- Supervisor-worker patterns
- Added complexity

### 22. Are Multiple AI Agents Better Than One?

- Specialization
- Parallel work
- Communication overhead
- Conflicting outputs
- Cost amplification
- Single-agent baseline comparison

### 23. What Is MCP, and Why Do AI Agents Need It?

- MCP hosts, clients, and servers
- Tools, resources, and prompts
- Capability discovery
- Standardized integration
- MCP vs ordinary APIs
- Why MCP is not memory

### 24. Why Are AI Agents Still Unreliable?

- Stochastic decisions
- Tool failure
- State drift
- Partial execution
- Poor recovery
- Missing observability
- Safety and permission gaps

## Cluster 3: Mathematics Behind AI

### 25. Why Does AI Need Mathematics?

- Data as numbers
- Probability
- Linear algebra
- Optimization
- Similarity
- Evaluation metrics

### 26. Why Does AI Represent Words as Numbers?

- Text cannot be processed directly by neural networks
- Tokens mapped to vectors
- Meaning represented through position and relationship
- Similar words ending up close together

### 27. What Is an Embedding?

- Dense vector representation
- Semantic similarity
- Embedding space
- Search and recommendation use cases
- Limits of embedding meaning

### 28. Why Does AI Use Vectors?

- Vectors as numerical representations
- Direction and distance
- Similarity search
- Model computation
- Visual intuition for vector space

### 29. How Does AI Measure Similarity?

- Dot product
- Cosine similarity
- Nearest neighbors
- Semantic search
- Why similar does not always mean correct

### 30. What Does Probability Have to Do with ChatGPT?

- Probability over next tokens
- Uncertainty
- Sampling
- Confidence vs correctness
- Why probable text can still be false

### 31. What Does Temperature Actually Change in an AI Model?

- Distribution flattening and sharpening
- Low vs high temperature
- Predictability vs variety
- Creative writing vs factual answers
- Interaction with top-p sampling

### 32. Why Are Matrices Everywhere in AI?

- Data batches
- Vector transformations
- Neural network layers
- Attention computations
- Hardware efficiency

### 33. How Does an AI Model Learn from Its Mistakes?

- Prediction error
- Loss functions
- Gradients
- Parameter updates
- Training vs inference

### 34. What Is Gradient Descent in Simple Words?

- Loss landscape
- Direction of improvement
- Step size
- Iterative updates
- Local minima and practical intuition

### 35. Why Does AI Need So Much Data?

- Pattern learning
- Coverage of language and concepts
- Generalization
- Data quality vs quantity
- Limits of scale

### 36. What Does It Mean to Train an AI Model?

- Training data
- Objective function
- Parameters
- Optimization
- Validation
- Difference between training, fine-tuning, and inference

## Cluster 4: AI, Work, and Education

### 37. Will AI Replace Software Engineers?

- Automation vs replacement
- Task-level impact
- Senior engineering judgment
- Code review and system design
- Changing skill expectations

### 38. Will Learning Programming Still Matter?

- Programming as system thinking
- Reading and evaluating generated code
- Debugging
- Architecture
- Tool-building ability

### 39. Which Parts of Software Development Can AI Automate?

- Boilerplate code
- Tests
- Documentation
- Refactoring assistance
- Code explanation
- Limits around architecture and accountability

### 40. Why Does AI Make Senior Engineering Skills More Important?

- Ambiguous requirements
- Reviewing AI output
- System trade-offs
- Debugging generated code
- Reliability and maintainability

### 41. Should Students Still Study Computer Science?

- Foundations that remain useful
- Algorithms and data structures
- Systems thinking
- Mathematics
- AI as an amplifier, not a substitute for understanding

### 42. What Mathematics Should You Learn for AI?

- Linear algebra
- Probability
- Calculus basics
- Optimization
- Statistics
- Practical learning order

### 43. Can Someone Without a Technical Background Learn AI?

- Conceptual entry points
- Practical limitations
- No-code and low-code tools
- When programming becomes necessary
- Suggested learning path

### 44. Will AI Make Entry-Level Jobs Disappear?

- Entry-level task automation
- Apprenticeship problem
- New expectations
- Portfolio and judgment signals
- Uncertainty across industries

### 45. What Should Children Learn in the Age of AI?

- Reading and writing clearly
- Mathematics
- Logical reasoning
- Curiosity
- Tool literacy
- Human judgment

### 46. Is Prompt Engineering a Long-Term Career?

- Prompting as a skill
- Prompting as interface design
- Shift toward systems and workflows
- Domain expertise
- Automation of simple prompt tasks

### 47. Can AI Really Make Everyone More Productive?

- Task suitability
- Quality control overhead
- Skill amplification
- Context switching
- When AI slows work down

### 48. Why Can Using AI Sometimes Make Work Slower?

- Reviewing bad output
- Prompt iteration
- Context setup
- Debugging generated mistakes
- Misfit between tool and task

## Cluster 5: Trust, Safety, and Privacy

### 49. Is It Safe to Share Personal Information with ChatGPT?

- Sensitive data
- Product and account settings
- Enterprise controls
- Data retention
- Practical sharing rules

### 50. Can Your Company Read What You Enter into an AI Tool?

- Workplace accounts
- Admin controls
- Logging
- Compliance policies
- Personal vs company tools

### 51. Can ChatGPT Reveal Someone Else's Information?

- Training data concerns
- Privacy safeguards
- Data leakage risks
- Hallucinated personal data
- Why secrets should not be shared

### 52. Why Should You Review AI-Generated Code?

- Incorrect logic
- Security issues
- Hidden assumptions
- Dependency and API mistakes
- Maintainability problems

### 53. What Is Prompt Injection?

- Instructions hidden in content
- Direct vs indirect prompt injection
- Tool-using agents
- Data exfiltration risks
- Defensive design basics

### 54. Can an AI Agent Be Tricked by a Webpage?

- Indirect prompt injection
- Untrusted web content
- Tool permissions
- Browser and retrieval agents
- Isolation and validation

### 55. How Do Companies Prevent AI from Taking Dangerous Actions?

- Permission scopes
- Human approval
- Policy checks
- Sandboxing
- Audit logs
- Monitoring

### 56. Who Is Responsible When an AI System Makes a Mistake?

- Human ownership
- Product and engineering responsibility
- Operational controls
- Review processes
- Legal and policy uncertainty

### 57. Can We Measure Whether an AI Answer Is Correct?

- Ground truth
- Human evaluation
- Automated checks
- Retrieval evidence
- LLM judges
- Limits of evaluation

### 58. Why Is Human Approval Still Necessary?

- Consequential actions
- Ambiguity
- Accountability
- Risk thresholds
- Escalation
- Human-in-the-loop design

### 59. Can AI Explain Why It Produced an Answer?

- Explanations vs actual internal process
- Reasoning traces
- Faithfulness problem
- Evidence-based explanations
- Limits of transparency

### 60. Should You Trust Citations Generated by AI?

- Fabricated citations
- Retrieval-backed citations
- Source checking
- Link rot
- Citation quality

## Cluster 6: AI Infrastructure

### 61. Where Does ChatGPT Actually Run?

- Data centers
- GPUs and accelerators
- Model serving
- Network requests
- Latency and scaling

### 62. Is an AI Data Center One Giant Computer?

- Clusters
- Servers
- GPUs
- Networking
- Distributed workloads
- Failure handling

### 63. What Happens Inside a GPU When AI Generates an Answer?

- Parallel computation
- Matrix operations
- Model weights
- Token-by-token inference
- Memory bandwidth

### 64. Why Does AI Need GPUs?

- Parallel math
- Matrix multiplication
- Training and inference
- Throughput
- CPU vs GPU intuition

### 65. Why Does AI Consume So Much Electricity?

- Large-scale computation
- Training cost
- Inference demand
- Data center cooling
- Efficiency improvements

### 66. What Is the Difference Between AI Training and Inference?

- Learning parameters vs using parameters
- Training data
- Model weights
- Serving user requests
- Cost and latency differences

### 67. Why Are AI Chips So Expensive?

- Specialized hardware
- Manufacturing complexity
- Memory and interconnects
- Demand
- Data center integration

### 68. Why Can ChatGPT Become Slow During Heavy Demand?

- Request queues
- GPU capacity
- Model size
- Rate limits
- Scaling constraints

### 69. How Is an AI Model Distributed Across Multiple Machines?

- Model parallelism
- Data parallelism
- Tensor parallelism
- Network communication
- Failure and synchronization

### 70. What Happens When a Machine Fails During AI Training?

- Checkpointing
- Restarting jobs
- Distributed coordination
- Lost work
- Fault tolerance

### 71. Why Does Running an AI Model Cost Money?

- Hardware
- Electricity
- Memory
- Serving infrastructure
- Engineering operations
- Token-based pricing

### 72. Can AI Models Run on Your Laptop?

- Small models
- Quantization
- Memory requirements
- Local inference
- Trade-offs vs hosted models

## Cluster 7: AI Images, Video, and Creativity

### 73. How Does AI Generate an Image from Words?

- Text prompt encoding
- Latent representation
- Diffusion intuition
- Denoising steps
- Image decoding

### 74. Does an AI Image Generator Copy Existing Images?

- Training on image-text pairs
- Pattern learning
- Memorization risk
- Style imitation
- Copyright concerns

### 75. Why Does AI Sometimes Draw Hands Incorrectly?

- Distributional learning
- Complex geometry
- Training data variation
- Local plausibility vs global structure
- Model improvements

### 76. How Can AI Create the Face of a Person Who Does Not Exist?

- Learned visual patterns
- Latent space
- Face structure
- Sampling variation
- Nonexistent identities

### 77. What Is a Diffusion Model?

- Noise
- Denoising
- Training objective
- Sampling process
- Text conditioning

### 78. How Does AI Generate Video?

- Frames and temporal consistency
- Motion modeling
- Diffusion or transformer approaches
- Prompt conditioning
- Common artifacts

### 79. Can AI Have Original Ideas?

- Recombination
- Novel outputs
- Human judgment
- Creativity definitions
- Limits of originality claims

### 80. Who Owns an AI-Generated Image?

- User rights
- Platform terms
- Copyright uncertainty
- Training data concerns
- Jurisdiction differences

### 81. How Can You Tell Whether an Image Was Generated by AI?

- Visual artifacts
- Metadata
- Watermarking
- Detection tools
- Limits of detection

### 82. Why Does the Same Prompt Produce Different Images?

- Random seeds
- Sampling
- Model settings
- Prompt ambiguity
- Variation as a feature

## Initial Season: How ChatGPT Works

### Season 1 Article Order

1. What Happens After You Send a Prompt to ChatGPT?
2. What Is a Token in AI?
3. Why Does AI Represent Words as Numbers?
4. What Is an Embedding?
5. How Does ChatGPT Choose Its Next Word?
6. What Does Temperature Actually Change in an AI Model?
7. Why Does ChatGPT Give Different Answers to the Same Question?
8. What Is a Context Window?
9. How Does ChatGPT Remember a Conversation?
10. Does ChatGPT Search the Internet for Every Answer?
11. Why Does ChatGPT Make Things Up?
12. Does ChatGPT Actually Understand Language?

### Season 1 Visuals

1. Prompt moving through the response-generation pipeline
2. Sentence breaking into tokens
3. Tokens becoming vectors
4. Word vectors changing with context
5. Probability distribution over next tokens
6. Temperature reshaping a probability distribution
7. Token-by-token generation
8. Context window filling and overflowing
9. Conversation history entering a new request
10. Retrieval bringing external information into the prompt
11. Unsupported fluent answer becoming a hallucination
12. Pattern recognition vs human understanding

## Explainer Article Template

- Natural-language question as title
- Short answer
- Visual explanation
- What actually happens
- Simple example
- One level deeper
- Common misconception
- Important limitation
- Continue learning

## Explainers Landing Page

- Featured explainer
- Start here sequence
- Browse by topic
- Latest explainers
- Go deeper links
