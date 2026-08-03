# Frontend — Atomic Design + pfusch

Read this for any frontend task involving components, layouts, styling, or browser-side behaviour.

---

## Atomic Design Hierarchy

Structure all FE components across five strict layers. **Layer boundaries are hard rules** — lower layers never import from higher ones:

```
atoms        ← primitive UI elements, no dependencies on other components
  ↑
molecules    ← simple groups of atoms with a single purpose
  ↑
organisms    ← complex sections composed of molecules and/or atoms
  ↑
templates    ← layout structure, no real data — defines content skeleton
  ↑
pages        ← concrete route instances, compose templates with real data/state
```

**Atoms**
- Smallest functional UI unit — a button, an input, a label, an icon.
- No knowledge of context, layout, or business logic.
- **Fully unstyled / headless**: atoms define structure and behaviour only. Visual styling is applied at molecule level and above.
- Cannot import any other component.

**Molecules**
- A small, focused group of atoms that does one thing well (e.g. a search form = label + input + button).
- This is where atoms first receive contextual styling.
- Single responsibility: if a molecule does two distinct things, split it.

**Organisms**
- Distinct, reusable sections of interface (e.g. a page header, a product grid, a comment thread).
- May compose molecules, atoms, and other organisms.
- Organisms are where business-domain concepts start to appear.

**Templates**
- Page-level layouts. Define where organisms and molecules are placed.
- Work with placeholder/skeleton content — no real data.
- Focus on content structure and spacing, not final content.

**Pages**
- Concrete instances of templates with real data wired in.
- Data fetching and state management lives here (or in dedicated hooks/loaders).
- Thin by design: a page should mostly compose and connect, not implement.

---

## Styling Rules

- **Co-locate styles with components by default**: CSS lives in the same file as the component.
- **Extract to a separate file when the component exceeds ~300 lines**: at that point a `.css` or `.module.css` file alongside the component is the right call. This is a sensible default, not a hard rule — use judgement.
- Never use global styles for component-specific concerns.

---

## pfusch (Current FE Stack — plain JS, no TypeScript)

The current FE stack uses [pfusch](https://github.com/MatthiasKainer/pfusch) — a tiny, browser-first custom elements library with no build step. Understand its model before writing any FE code.

### Core mechanics
- Components are defined with `pfusch(tagName, initialState, (state) => [...elements])`.
- State is a `Proxy` — assign exact declared top-level keys directly (`state.count++`) and use `state.mutate(...)` for synchronous nested changes.
- Templates return lightweight `html.*` descriptors, real DOM nodes, strings, static `css` descriptors, `script(...)` descriptors, nested arrays, or falsy entries. There is no virtual DOM.
- `script()` blocks run once per connected lifecycle, before template-owned nodes are synchronized. Return cleanup for external listeners and subscriptions; reconnecting runs the script again.
- Shadow DOM is used — styles are scoped, slots are explicit.
- Keep `css` template text static because compiled stylesheets are held in a global, unbounded cache. Use CSS variables or classes for dynamic values.
- `helpers.children()` returns the original Light DOM nodes; `helpers.childElements()` returns descriptor-wrapped versions suitable for templates.

### Component communication — events first
- `trigger(name, detail)` dispatches both `component-name.name` and bare `name` bubbling, composed `CustomEvent`s from the component.
- It also broadcasts a namespaced `window.postMessage` whose `detail.data` is JSON text. Pass plain serializable data, never a raw DOM event or complex library object.
- Never create direct references between components — keep them loosely coupled through events.

### Progressive enhancement
- Write HTML first. The baseline should be useful without JavaScript.
- Enhance with pfusch for client-side behaviour — don't rescue a blank page.
- Good candidates: forms, tables with filtering/sorting, tabs, dashboards with fallback content.
- Bad candidates: UI with no meaningful HTML baseline, canvas-heavy graphics.

### Testing pfusch components
- Use `pfuschTest` + `setupDomStubs()` — no browser required.
- Pattern: mount → flush → interact → flush → assert on rendered output (not internal state).
- Assert on what the user sees (`textContent`, DOM structure) over asserting on `state.*`.
- `pfuschTest()` and `.get()` return `PfuschNodeCollection`; prefer `.first`, `.at()`, `.click()`, `.submit()`, `.value`, `.checked`, `.textContent`, attribute helpers, and `.flush()` over indexing `.elements`.

### Atomic Design mapping in pfusch
- Atoms → single `pfusch()` components with no child component dependencies.
- Molecules → `pfusch()` components that compose atoms via `html.slot()` or direct children.
- Organisms → pfusch components that wire molecules together and own event subscriptions.
- Templates → plain HTML layout files that declare the skeleton, mount organisms via custom element tags.
- Pages → HTML entry points that assemble templates with real data sources.

### Plain JS rules (no TypeScript here)
- The no-`any` rule doesn't apply, but all other non-negotiables do.
- Use JSDoc comments on exported component definitions to document props/state shape.
- Named constants for all event name strings — never inline magic strings for event names.
