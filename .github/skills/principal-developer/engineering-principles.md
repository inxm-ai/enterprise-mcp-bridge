# Engineering Principles

Language-agnostic fundamentals that apply to every piece of code regardless of layer or stack.

---

## Error Handling — Railway Oriented Programming

Model errors as data. Never throw/raise for expected failure paths. The pattern adapts per language — the two-track principle (happy path / error path, always explicit) is universal.

### Per-language implementation

**TypeScript** — custom domain error types + `Result<T, E>`:
```typescript
type Ok<T> = { readonly ok: true; readonly value: T };
type Err<E> = { readonly ok: false; readonly error: E };
type Result<T, E> = Ok<T> | Err<E>;

const ok = <T>(value: T): Ok<T> => ({ ok: true, value });
const err = <E>(error: E): Err<E> => ({ ok: false, error });
```
- Functions that can fail return `Result<T, E>`, not `T | null` or `T | undefined`.
- When you care about the error → wrap in a named domain error type (`PaymentDeclinedError` beats a raw string). Types are free — use them.
- When you don't care → return `null` / `undefined`. The absence is the signal.

**Plain JavaScript** — defensive, no additional structure:
```javascript
// ❌ Wrong — throws blow up the page
function parseConfig(raw) {
  if (!raw.url) throw new Error('missing url');
}

// ✅ Right — be defensive, return or don't
function parseConfig(raw) {
  try {
    return JSON.parse(raw);
  } catch {
    return null;
  }
}
```
- Be defensive: guard inputs, fail gracefully, return `null` / `undefined` / nothing on failure.
- No `throw`. No custom `Error` subclasses. No `Result`-like objects. No additional structure.
- Don't blow up the page. Don't rethrow. Just return or don't.

**Python** — `returns` library at domain boundaries:
```python
from returns.result import Result, Success, Failure

def parse_user(data: dict) -> Result[User, str]:
    ...
```
- Never bare `except:`. Always catch specific exception types and convert to `Failure` at boundaries.
- When you care → `Failure(meaningful_description)`.
- When you don't → return `None`.
- **MCP bridge**: `try/except` is sometimes unavoidable due to the bridge runtime. Still never rethrow raw — convert to `Failure` or swallow deliberately with a comment explaining why.

**Rust** — idiomatic `Result<T, E>` + `?` operator. `thiserror` for domain error types, `anyhow` for application-level aggregation. Unchanged — the language enforces this naturally.

### Universal rules
- Never swallow errors silently (`catch {}`, `except: pass`, ignoring a `null` return).
- Log at the boundary where you *handle* the error, not where you create it.
- Programmer errors / invariant violations (things that should never happen) → panic/throw is acceptable. Expected failures → always modelled as data.
- Think in two tracks: happy path and error path. Keep them explicit in the code structure.

---

## Functional-First Style

Default to functions and data transformation pipelines. Reach for classes/objects only when:
- You need to encapsulate mutable state with a clear lifecycle (e.g. a connection pool, a state machine).
- You're implementing a well-known OOP pattern that genuinely simplifies the design.
- The language/framework idiom demands it.

Prefer:
- Pure functions with no side effects.
- Composition over inheritance.
- Small, single-purpose functions with descriptive names.
- `map`, `filter`, `reduce` over imperative loops where clarity is equal or better.

---

## Comments — "Why", Never "What"

```typescript
// ❌ Wrong: explains what the code does (already obvious)
// Increment counter
counter++;

// ✅ Right: explains why a non-obvious decision was made
// We increment before the async call to prevent a race condition where
// two concurrent requests both read count=0 and both proceed.
counter++;
```

JSDoc/docstrings on **all public API surfaces** (exported functions, public class methods, module entry points). Include param descriptions and return type rationale when non-obvious.

No commented-out code in final output. Use `TODO:` or `FIXME:` with a brief explanation if something is intentionally incomplete.

---

## Code Structure

**Feature/domain-based folder structure.** Group by what the code *is about*, not what type of file it is.

```
// ❌ Wrong: technical layering
src/
  controllers/
  services/
  models/
  utils/

// ✅ Right: domain-based
src/
  users/
    user.types.ts
    user.service.ts
    user.service.test.ts
    user.repository.ts
  orders/
    order.types.ts
    order.service.ts
    ...
  shared/
    result.ts
    constants.ts
```

Shared utilities go in `shared/` or `common/`. Keep nesting shallow — max 3 levels deep is a strong signal to reconsider the structure.

---

## Dependencies

Pragmatic — use well-maintained libraries freely. Don't reinvent the wheel.

**Preferences by ecosystem:**
- **TypeScript/JS**: `zod` for runtime validation, `vitest` or `jest` for testing, `neverthrow` or custom `Result` for error handling.
- **Python**: `pydantic` for data validation, `pytest` for testing, `returns` for ROP-style error handling.
- **Rust**: `thiserror` + `anyhow`, `tokio` for async, `serde` for serialisation.

When introducing a dependency, briefly note why — especially if non-obvious.

---

## YAGNI — Full Detail

Never build a capability because you *presume* you'll need it in the future. Build it when it is actually needed.

Every presumptive feature carries three costs:
- **Cost of build**: effort on something that may never be used. Analysis at Microsoft found roughly ⅔ of carefully-planned features don't improve the metrics they were designed to improve.
- **Cost of delay**: while building the presumptive feature, something with real current value wasn't built.
- **Cost of carry**: the extra code adds complexity that slows down everything else indefinitely.

In practice:
- Before building something speculative, imagine the refactoring needed to add it later. It is almost always cheaper than the carry cost of building it now.
- Something cheap that meaningfully reduces *future* cost with minimal complexity today (e.g. a named constant, a clean interface boundary) is acceptable.
- Any extensibility point that is never used isn't just wasted effort — it actively gets in the way.
- When in doubt: don't build it.

YAGNI applies equally to LLM-generated code. The ease of generating large volumes of code with AI makes speculative building *more* tempting. Hold the same standard regardless of who (or what) writes the code.

---

## Thin Vertical Slices

Deliver every feature as the thinnest possible slice through the full stack that produces real, observable value — from UI (if applicable) down through domain logic to persistence/infrastructure.

**Why slices, not layers:**
Building the entire data layer first, then the service layer, then the UI means nothing is demonstrably working until the end. A thin vertical slice is working — testable, deployable, shippable — from day one.

**What thin means:**
- Implement exactly the happy path needed to satisfy the current requirement. No edge cases that aren't yet required, no generalisation that isn't yet needed (YAGNI).
- A slice should be completable in a single PR. If it isn't, it's not thin enough — split it.
- Incomplete slices live behind a feature flag, not in a long-lived branch.

**In practice:**
- When given a feature to build, identify the thinnest path from input to output that delivers observable value.
- Propose that slice explicitly before writing code: "Here's the slice I'm implementing — [description]. This excludes [X, Y] which would be follow-up slices."
- Each slice gets its own tests, its own instrumentation, and leaves the codebase deployable.
- Resist the pull toward "let me just also handle..." — that's the next slice.

This works directly with evolutionary architecture: each slice is a validated hypothesis. The architecture grows from real slices, not anticipated ones.

---

## Refactoring (Martin Fowler)

### Core Practices

- **Small, safe steps**: each refactor must leave the test suite green. Never refactor and change behaviour in the same step — if the tests break, the step was too large.
- **Separate refactoring from feature work**: refactoring commits and feature commits must not be mixed. A commit either changes behaviour (feature/fix) or improves structure (refactor) — never both.
- **Refactor freely around the feature**: fine to refactor before *or* after adding a feature — the key is keeping the two concerns in separate commits.
- **Rule of Three for abstractions**: do not propose extracting an abstraction until a pattern appears in **3 or more distinct locations**. One = leave it. Two = note it. Three = propose it. Premature abstraction is worse than duplication.

### Code Smells — Actively Flag These

When working on any code task, scan for these smells. **Do not silently fix them** — flag in the tradeoffs section and offer to address separately:

**Primitive Obsession**
Raw primitives (`string`, `number`, `boolean`) used where a domain type should exist:
```typescript
// ❌ Smell
function sendEmail(email: string, userId: string, role: string) { ... }

// ✅ Better
function sendEmail(email: EmailAddress, userId: UserId, role: UserRole) { ... }
```

**Feature Envy**
A function that seems more interested in another module's data than its own. If logic reaches deeply into another object's internals, it probably belongs closer to that data.

**Data Clumps**
The same group of 3+ fields or parameters appearing together repeatedly is a type waiting to be born:
```typescript
// ❌ Smell: these three always travel together
function createUser(firstName: string, lastName: string, email: string) { ... }
function updateUser(id: UserId, firstName: string, lastName: string, email: string) { ... }

// ✅ Better
type UserIdentity = { readonly firstName: string; readonly lastName: string; readonly email: EmailAddress };
```

**Divergent Change / Shotgun Surgery**
- *Divergent Change*: one module changes for many different reasons (low cohesion).
- *Shotgun Surgery*: one change requires edits scattered across many modules (high coupling).
Both flagged in tradeoffs with a suggested structural remedy — not silently fixed.

### What Not To Do
- Do not extract an abstraction speculatively. Inline duplication is preferable to the wrong abstraction.
- Do not refactor code that has no tests — stabilise with tests first, then refactor.
- Do not rename things "while you're in there" during a feature commit — schedule as a separate step.
