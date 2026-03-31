---
description: "Instruction file writing standards. Covers density, structure, and adversarial review."
applyTo: "**/*instructions*.md"
---
# Writing Agent Instructions

## Principles

1. **Density** — every line earns its place; if removable without loss, delete it
2. **No duplication** — don't repeat what code, types, or names already say
3. **Encode the unrecoverable** — conventions, non-obvious "why", correctness constraints
4. **Task-flow order** — most frequent needs first

## Format

- `description` required in frontmatter (discovery surface); `applyTo` for file-pattern triggers
- Tables for tool/pattern mappings, not prose lists
- Imperative rules ("Never X"), not hedged suggestions ("Consider...", "You may want to...")
- Examples only when the pattern isn't obvious; inline over blocks

## Adversarial Review

Run three passes, iterate until stable:

**Line-level**: Would an agent err without this line? Can it be inferred from code? Duplicated elsewhere? → delete or move.

**Structure-level**: Does section order follow task flow? Any redundant meta-layers? Anything misplaced?

**Adversarial**: What would confuse a new agent? What's missing that causes errors? What's verbose that could be a table row?

## Section Order (project instructions)

Header (module map, entry point, instruction registry) → Workflow → Code Style → Domain → Roadmap.

The instruction file registry table is the primary discovery mechanism — keep it in the Header.

## Anti-patterns

- Abstract prose before concrete rules (flip order)
- Explaining tool benefits (agent already knows its tools)
- Stating something twice: "Package manager: pip" + "`pip install`" (latter implies former)
- Headers without actionable content under them
