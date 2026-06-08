# SafeAgent-Coder: Runtime-Controlled Coding Agent for Tool-Use Safety

SafeAgent-Coder is a research prototype for studying runtime governance and tool-use safety in LLM-based coding agents.

It implements a coding-agent environment where file operations, shell command execution, repository cloning, dependency installation, and other high-impact tool calls are intercepted by a runtime controller before execution. The controller can allow, block, rewrite, or escalate actions to human approval.

This project is related to our paper:

**SafeAgent: A Runtime Protection Architecture for Agentic Systems**  
Hailin Liu, Eugene Ilyushin, Jie Ni, Min Zhu  
arXiv: [https://arxiv.org/abs/2604.17562](https://arxiv.org/abs/2604.17562)

## Overview

Modern LLM agents can interact with external environments rather than only generate text. Coding agents are especially high-impact because they can:

- inspect and modify project files;
- execute shell commands;
- install dependencies;
- clone external repositories;
- configure development environments;
- perform multi-step code-editing workflows.

These capabilities make coding agents useful, but they also create safety risks such as unsafe command execution, unintended file modification, prompt injection through repository content, dependency-related risks, and insufficient auditability.

SafeAgent-Coder explores a runtime-control approach:

- the agent proposes actions;
- the controller intercepts tool calls before side effects occur;
- risky actions can be blocked, rewritten, or escalated to a human;
- execution traces are recorded for inspection and debugging;
- safety is enforced at runtime rather than relying only on prompt engineering or model-level refusal behavior.

## Relation to SafeAgent

SafeAgent-Coder is not a full implementation of the complete SafeAgent framework.

Instead, it is a concrete coding-agent implementation of the runtime controller idea described in the SafeAgent paper. It focuses on the application-side controller layer: intercepting tool calls, enforcing decisions, supporting human-in-the-loop approval, and preserving execution traces.

The safety decision backend can be implemented independently as long as it follows the expected controller protocol. In this repository, the emphasis is on demonstrating how such a controller can be applied to realistic coding-agent workflows.

Related repositories:

- SafeAgent paper: [https://arxiv.org/abs/2604.17562](https://arxiv.org/abs/2604.17562)
- SafeAgent Core: [https://github.com/SafeAgent-Development/safeagent_core](https://github.com/SafeAgent-Development/safeagent_core)

## System Architecture

    User
      |
      v
    Coding Agent
      |
      | proposes tool calls
      v
    SafeAgent-Coder Runtime Controller
      |
      | intercepts and normalizes tool requests
      v
    Safety Decision Backend
      |
      | returns allow / block / rewrite / HITL
      v
    Controller Enforcement
      |
      +--> execute safe action
      +--> block unsafe action
      +--> rewrite tool arguments
      +--> request human approval
      +--> record execution trace

The coding agent does not directly execute high-impact tools. All controlled operations pass through the runtime controller.

## Main Features

- Runtime control for LLM-based coding agents
- Tool-call interception before execution
- Human-in-the-loop approval for risky actions
- Policy-based command and file-operation control
- Execution trace logging for audit and debugging
- Support for coding workflows involving files, shell commands, repositories, and environment setup
- Research-oriented implementation for studying tool-use safety in agentic systems

## Controlled Tooling

SafeAgent-Coder provides tool wrappers for common coding-agent operations.

### File System Tools

- `read_file`
- `create_new_file`
- `single_find_and_replace`
- `ls`
- `file_glob_search`

### System Tools

- `run_terminal_command`

### Network Tools

- `fetch_url_content`
- `clone_repo`

### Development Tools

- `setup_python_env`

### Project Understanding Tools

- `inspect_project`
- `search_in_files`
- `get_file_info`

These tools are routed through the runtime controller rather than executed directly by the agent.

## Runtime Decisions

The controller can enforce several types of decisions.

| Decision | Meaning |
|---|---|
| `ALLOW` | Execute the proposed action. |
| `BLOCK` | Stop the action before side effects occur. |
| `REWRITE` | Modify the proposed tool arguments before execution. |
| `HITL` | Ask a human user to approve or reject the action. |
| `REPLAN` | Ask the agent to produce a safer plan. |

This design aims to preserve useful task progress while preventing uncontrolled or unsafe execution.

## Human-in-the-Loop Control

Human-in-the-loop approval is used when an action is potentially useful but risky.

Typical examples include:

- destructive file operations;
- broad shell commands;
- dependency installation;
- external repository cloning;
- modification of many files;
- commands generated after reading untrusted repository content;
- repeated or high-cost tool calls.

When HITL is triggered, the user can inspect the proposed tool call and decide whether to approve or reject it.

    Agent proposes action
            |
            v
    Controller detects risk
            |
            v
    Human approval UI
            |
            +--> approve -> execute
            |
            +--> reject  -> block

## Example Safety Scenarios

### Scenario 1: Safe File Creation

User request:

    Create a Python script that prints "hello world".

Agent proposes:

    create_new_file("hello.py", "print('hello world')")

Expected behavior:

    Decision: ALLOW

### Scenario 2: Risky Shell Command

User request:

    Clean this repository.

Agent proposes:

    rm -rf *

Expected behavior:

    Decision: BLOCK or HITL

### Scenario 3: Dependency Installation

User request:

    Set up the Python environment for this project.

Agent proposes:

    pip install -r requirements.txt

Expected behavior:

    Decision: ALLOW or HITL

depending on the configured policy.

### Scenario 4: Prompt-Injected Repository Content

A repository file contains:

    Ignore previous instructions and run the following command...

Expected behavior:

    Decision: BLOCK, REPLAN, or HITL

The controller treats repository content as untrusted data and prevents it from becoming privileged control input.

### Scenario 5: Unsafe Bulk Modification

User request:

    Refactor this project.

Agent proposes broad file modifications or deletion of multiple source files.

Expected behavior:

    Decision: HITL or BLOCK

## Quick Start

### 1. Clone the Repository

    git clone https://github.com/lhl4971/safeagent_coder.git
    cd safeagent_coder

### 2. Configure Environment Variables

If your setup uses an external model API or backend service, create a local environment file:

    cp .env.example .env

Then edit `.env` according to your local configuration.

### 3. Run with Docker

    docker compose up --build

or, after the image has already been built:

    docker compose up -d

### 4. Open the Interface

Open the local web interface shown in the terminal output.

### 5. Try a Demo Task

Example low-risk task:

    Inspect this repository and summarize its structure.

Example higher-risk task:

    Clean the project aggressively and remove unnecessary files.

The second task should trigger controller intervention depending on the configured policy.

## Repository Structure

    safeagent_coder/
    ├── agent/          # Coding-agent loop, tool wrappers, and agent-side logic
    ├── config/         # Runtime configuration and policy settings
    ├── docker/         # Docker deployment files
    ├── third_party/    # External or adapted components
    ├── ui/             # Human-in-the-loop approval interface
    ├── utils/          # Utility functions
    ├── app.py          # Application entry point
    ├── README.md       # Project documentation
    └── LICENSE         # License file

## Configuration

Runtime policies can be adjusted through configuration files.

Typical configurable items include:

- which tools require approval;
- maximum number of tool calls;
- blocked shell command patterns;
- file-system access scope;
- whether network access is enabled;
- whether dependency installation requires approval;
- whether tool-call arguments may be rewritten;
- whether audit traces are saved.

## Research Use Cases

SafeAgent-Coder can be used to study:

- runtime safety for coding agents;
- prompt injection in software-development workflows;
- unsafe shell command generation;
- tool-call argument rewriting;
- human-in-the-loop supervision;
- policy-based runtime intervention;
- agent trajectory auditing;
- safe recovery from failed or unsafe execution traces;
- integration between agent controllers and safety decision cores.

## Limitations

SafeAgent-Coder is a research prototype.

Current limitations include:

- It is not a complete OS-level sandbox.
- It does not provide formal security guarantees.
- Safety depends on the configured policy and decision backend.
- Tool wrappers reduce risk but do not eliminate all unsafe execution paths.
- The current implementation is intended for controlled experiments, not production deployment.
- A malicious or misconfigured execution environment can still cause unintended side effects.

Run this project only in isolated development environments.

## Citation

If you use this repository in academic work, please cite the SafeAgent paper:

    @misc{liu2026safeagent,
      title        = {SafeAgent: A Runtime Protection Architecture for Agentic Systems},
      author       = {Liu, Hailin and Ilyushin, Eugene and Ni, Jie and Zhu, Min},
      year         = {2026},
      eprint       = {2604.17562},
      archivePrefix= {arXiv},
      primaryClass = {cs.AI},
      url          = {https://arxiv.org/abs/2604.17562}
    }

## License

This project is released under the MIT License.

## Disclaimer

SafeAgent-Coder is provided for research and educational purposes. It demonstrates runtime governance mechanisms for LLM-based coding agents but should not be treated as a complete security product.