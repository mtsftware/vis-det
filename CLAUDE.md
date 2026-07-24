# Developer AI Guide

## 🛑 CRITICAL AGENT CONSTRAINTS (COMPUTE & TOKEN LIMITS)
You MUST STRICTLY adhere to the following constraints. Do not attempt to bypass them:
- **NO PACKAGE INSTALLATIONS:** Do NOT install any modules, libraries, or dependencies (e.g., `apt`, `pip`, `npm`) inside the container.
- **NO INTERNET SEARCHES:** Do NOT use web search tools to look up documentation, forums, or external code.
- **NO AUTO-BUILDING:** Do NOT execute any build commands (`make`, `cmake`, `gcc`, etc.).
- **NO AUTO-TESTING:** Do NOT execute the compiled binaries, run test scripts, or attempt to validate the code by running it.
- *Your sole responsibility is to analyze the provided files, reason about the architecture, and generate necessary code.*
-**Don't use git commands**
