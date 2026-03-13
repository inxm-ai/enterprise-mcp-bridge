# Testing

Testing pyramid, mocks philosophy, and structural patterns for writing tests that stay green through refactoring.

---

## Testing Pyramid

```
         /\
        /E2E\          ← few, slow, high-confidence
       /------\
      / Integ  \       ← moderate, test real component boundaries
     /----------\
    /    Unit    \     ← many, fast, test business logic
   /--------------\
```

**Core rules:**
- Tests are written **from a business/behaviour perspective**, not an implementation perspective.
- Test *what* the code does, not *how* it does it internally.
- Avoid testing implementation details (private methods, internal state) — this makes tests brittle.
- Name tests in plain language: `it('returns an error when the user email is already taken')`.
- Prefer TDD: write the failing test first, then implement.
- **Integration tests**: test real interactions between components (e.g. service + DB).
- **E2E tests**: few, cover critical user journeys only.
- Avoid testing the same thing at multiple pyramid levels — no test duplication.

---

## Testing Without Mocks (James Shore)

**Avoid mocks by default.** Mocks lock in implementation details, make refactoring painful, and produce tests that verify nothing real. The preferred alternatives are:

### 1. State-Based Tests — always

Assert on *output and observable state*, never on *which methods were called*:

```javascript
// ❌ Interaction-based (locks in implementation, brittle)
expect(emailService.send).toHaveBeenCalledWith('welcome', user.email);

// ✅ State-based (tests what actually happened)
const result = await registerUser({ email: 'a@b.com' });
assert.equal(result.status, 'registered');
assert.deepEqual(outbox.data, [{ type: 'welcome', to: 'a@b.com' }]);
```

### 2. Logic Sandwich — separate pure logic from I/O

Keep business logic in pure functions with no I/O dependencies. Push all I/O (DB, HTTP, filesystem, clock) to the edges. Test logic independently with plain inputs and outputs — no infrastructure needed:

```javascript
// ✅ Pure logic — trivially testable, no infrastructure needed
const { nextState, commands } = processOrder(currentState, event);

// Infrastructure at the edges executes the commands
await Promise.all(commands.map(cmd => infrastructure.execute(cmd)));
```

### 3. Infrastructure Wrappers — own your boundaries

Never call 3rd party libraries or I/O directly from business logic. Always wrap them in a thin owned interface. This gives you a seam to substitute behaviour in tests, a single place to change when a 3rd party API changes, and a clear boundary for Nullable creation:

```javascript
// ❌ Direct 3rd party call — couples logic to library, hard to test
import { S3Client } from '@aws-sdk/client-s3';
async function saveReport(data) {
  const client = new S3Client({});
  await client.send(new PutObjectCommand({ ... }));
}

// ✅ Owned wrapper — substitutable, single seam
class ReportStore {
  static create() { return new ReportStore(new S3Client({})); }
  static createNull() { return new ReportStore(new NullS3Client()); }
  async save(report) { ... }
}
```

The `createNull()` factory returns a version that records calls and returns configurable responses without touching real infrastructure. Tests use `createNull()`; production uses `create()`.

### When mocks ARE acceptable

Mocks are acceptable when wrapping a 3rd party library in a full Infrastructure Wrapper + Nullable would be excessive relative to the value of the test — e.g. a thin adapter for a well-understood external SDK used in only one place. In this case, mock at the wrapper boundary only, never deep inside business logic. Always note in the tradeoffs section why a mock was chosen over a Nullable.

### What to never do
- Never mock your own domain objects or services — use the real thing or a Nullable.
- Never write tests that only verify mocks call other mocks — that tests nothing real.
- Never assert on *how* something was done internally — only on *what* the observable outcome was.

---

## Default Expectation

When writing code, always include at minimum the unit test(s) for the core logic unless the user explicitly says not to.

---

## pfusch Component Testing

- Use `pfuschTest` + `setupDomStubs()` — no browser required.
- Pattern: mount → flush → interact → flush → assert on rendered output.
- Assert on what the user sees (`textContent`, DOM structure) — not on `state.*`.
