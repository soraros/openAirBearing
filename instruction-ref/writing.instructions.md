---
description: "Agent instruction writing standards. Covers density, structure, and adversarial review."
applyTo: "**/*instructions*.md"
---
# Writing Agent Instructions

## Core Principles

1. **Information density** — each line must have purpose; if removable without loss, delete it
2. **No duplication** — don't repeat what's inferable from code, types, or names
3. **Encode the unrecoverable** — conventions, non-obvious "why", correctness-critical constraints
4. **Structure follows task flow** — most frequent needs first

## Format Guidelines

- **Frontmatter**: `description` required for discovery; `applyTo` for file-pattern matching
- **Tool mappings**: tables, not prose lists ("X over Y" → table row)
- **Behavioral rules**: explicit, imperative ("Never X", "Always Y")
- **Examples**: only when pattern isn't obvious; inline over blocks
- **Parentheticals**: avoid or promote to own line

## Review Process (adversarial iteration)

Act as a **critic in a GAN** — your goal is to find flaws. Run these passes:

**Pass 1 — Line-level**: For each line, ask:
1. Would an agent make a mistake without this? → keep
2. Can this be inferred from code/types? → delete
3. Is this duplicated elsewhere? → delete
4. Does it belong in a different section? → move

**Pass 2 — Structure-level**: Evaluate overall document:
1. Does section order follow task flow? (most frequent needs first)
2. Are there redundant meta-layers? (abstract principle + concrete list saying same thing)
3. Is anything misplaced? (entry point in header vs workflow)

**Pass 3 — Adversarial**: Try to break it:
1. What would confuse a new agent?
2. What's missing that would cause errors?
3. What's verbose that could be a table row?

Iterate until no changes improve the document. Be harsh.

## Section Order (for project instructions)

| Section | Purpose |
|---------|---------|
| Header | Orient: what is this, module map, entry point, instruction file registry |
| Workflow | How to run/develop (build, tools, env, git rules) |
| Code Style | How to write matching code |
| Domain | Physics/math that can't be inferred from code |
| Roadmap | Future work (signals "don't implement") |

The instruction file registry (table of `*.instructions.md` files with scope and `applyTo` triggers) is the primary discovery mechanism. Keep it in the Header of `copilot-instructions.md`.

## Anti-patterns

- Abstract prose before concrete rules (flip order)
- "Consider..." / "You may want to..." (be imperative)
- Explaining tool benefits (agent already knows its tools)
- Verbose examples where a phrase suffices
- Stating something twice: "Package manager: uv" + "`uv sync`" (latter implies former)
- Headers without actionable content under them
