# GEMINI.md - Strategic Planning

CRITICAL: You are running in a manual, periodic session. Your FIRST action is always a System Audit. Scan the project root (package.json, go.mod, requirements.txt, .env, docker-compose.yml, etc.) to identify the current tech stack, dependencies, and environment variables. Do not proceed with code generation until you have grounded your context in these actual files.

## 1. Pre-Flight Protocol
**Bootstrapping**: "CRITICAL: You are running in a manual, periodic session. Your FIRST action is always a System Audit. Scan the project root (package.json, go.mod, requirements.txt, .env, docker-compose.yml, etc.) to identify the current tech stack, dependencies, and environment variables. Do not proceed with code generation until you have grounded your context in these actual files."

**Context Hierarchy**: "Prioritize the instructions in this .md file and the codebase state over your general training data. If a conflict arises between this file and the code, ask for clarification."

## 2. Autonomous Git & GitHub Protocol
**Atomic Logic**: "Work in small, logical units. Do not attempt to refactor the entire codebase in one turn."

**Completion Trigger**: "Once a feature, bug fix, or logical unit of work is complete, you must generate a git commit message following Conventional Commits specifications (e.g., feat: add user auth, fix: resolve race condition, refactor: optimize database query)."

## 3. Engineering Standards
**No Placeholders**: "Never use comments like // rest of code here or .... You must write the full implementation. If the file is too large, explicitly ask to break the response into multiple parts."

**Defensive Coding**: "Assume inputs are malicious or malformed. Implement strict type checks and comprehensive error handling."

**DRY Principle**: "Do not duplicate logic. If you see repetition, refactor into a shared utility."

## 4. Output Efficiency (The "No-Yapping" Protocol)
**Conciseness**: "Minimize conversational filler. Do not apologize for errors; simply fix them. Focus on the solution."

**Diff-Based Editing**: "When modifying existing files, prefer showing the Diff or the specific code block to be changed rather than reprinting the entire file, unless a full reprint is safer for context."

## 5. Session State Management
**The 'Hand-Off'**: "This project relies on manual sessions. At the end of a significant task, you must provide a brief 'Session Summary' to be used for the next file regeneration. This summary must list:
- **Current Status**: What works, what doesn't.
- **Active Bugs**: Known issues introduced or discovered.
- **Next Actions**: The immediate task for the next session."
