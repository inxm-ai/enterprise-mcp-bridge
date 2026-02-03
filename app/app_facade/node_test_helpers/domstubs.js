import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { createHash } from 'node:crypto';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { match } from 'node:assert';

const original = {
  window: globalThis.window,
  document: globalThis.document,
  Node: globalThis.Node,
  customElements: globalThis.customElements,
  HTMLElement: globalThis.HTMLElement,
  FormData: globalThis.FormData,
  CustomEvent: globalThis.CustomEvent,
  requestAnimationFrame: globalThis.requestAnimationFrame,
  CSSStyleSheet: globalThis.CSSStyleSheet,
  fetch: globalThis.fetch,
  localStorage: globalThis.localStorage,
  sessionStorage: globalThis.sessionStorage,
  KeyboardEvent: globalThis.KeyboardEvent,
  matchMedia: globalThis.matchMedia,
  MutationObserver: globalThis.MutationObserver,
  triggerMutation: globalThis.triggerMutation,
};
class FakeClassList {
  constructor() {
    this._set = new Set();
  }
  add(...names) {
    names.forEach(name => this._set.add(name));
  }
  remove(...names) {
    names.forEach(name => this._set.delete(name));
  }
  toggle(name, force) {
    if (force === undefined) {
      return this._set.has(name) ? (this._set.delete(name), false) : (this._set.add(name), true);
    }
    if (force) this._set.add(name);
    else this._set.delete(name);
    return force;
  }
  contains(name) {
    return this._set.has(name);
  }
  forEach(cb) {
    this._set.forEach(cb);
  }
  get length() {
    return this._set.size;
  }
  toString() {
    return Array.from(this._set).join(' ');
  }
}

class FakeAttributes extends Map {
  [Symbol.iterator]() {
    return Array.from(super.entries()).map(([name, value]) => ({ name, value }))[Symbol.iterator]();
  }
}

let suspendConnect = false;
const connectTree = (node) => {
  if (suspendConnect || !node) return;
  const walk = (current) => {
    if (!current) return;
    if (typeof current.connectedCallback === 'function') current.connectedCallback();
    if (current.childNodes) current.childNodes.forEach(child => walk(child));
    if (current.shadowRoot?.childNodes) current.shadowRoot.childNodes.forEach(child => walk(child));
  };
  walk(node);
};

const matchSelector = (el, selector) => {
  if (!selector || !el) return false;
  const normalized = selector.trim();
  if (!normalized) return false;

  const selectors = normalized.split(',').map(s => s.trim()).filter(Boolean);
  const matchesSingle = (sel) => {
    let base = sel;
    let attr = null;
    const attrMatch = sel.match(/^(.*?)\[([^=\]]+)=\s*"?([^\]"]+)"?\s*\]$/);
    if (attrMatch) {
      base = attrMatch[1] || '*';
      attr = { name: attrMatch[2], value: attrMatch[3] };
    }

    const baseParts = base.split('.').filter(Boolean);
    let tag = baseParts[0] || base;
    const classes = baseParts.slice(1);

    if (base.startsWith('.')) {
      tag = '*';
      classes.push(base.replace('.', ''));
    } else if (base.startsWith('#')) {
      return el.id === base.slice(1);
    }

    const customMatchOnly = !base.startsWith('.') && !base.startsWith('#') && base.includes('-') && globalThis.customElements?.get?.(base);
    const tagMatch = el.tagName?.toLowerCase() === tag.toLowerCase();
    const classMatch = el.classList.contains(tag);
    const idMatch = el.id === tag;
    const tagMatches = tag === '*'
      ? true
      : customMatchOnly
        ? tagMatch
        : (tagMatch || classMatch || idMatch);

    if (!tagMatches) return false;
    if (classes.length && !classes.every(cls => el.classList.contains(cls))) return false;
    if (attr) {
      const val = el.getAttribute(attr.name);
      if (val == null || String(val) !== String(attr.value)) return false;
    }
    return true;
  };

  return selectors.some(matchesSingle);
};

class FakeElement {
  constructor(tagName, ownerDocument) {
    this.tagName = String(tagName || '').toUpperCase();
    this.nodeType = 1;
    this.ownerDocument = ownerDocument;
    this._childNodes = [];
    this.dataset = {};
    this.state = {};
    this.classList = new FakeClassList();
    this.attributes = new FakeAttributes();
    this._listeners = new Map();
    this._innerHTML = '';
    this._textContent = '';
    this._value = '';
    this._checked = false;
    this._name = '';
    this._type = '';
    this._action = '';
  }
  get className() {
    return this.classList.toString();
  }
  set className(value) {
    this.classList = new FakeClassList();
    const names = String(value ?? '').split(/\s+/).filter(Boolean);
    this.classList.add(...names);
  }
  attachInternals() {
    if (!this._internals) {
      this._internals = {
        _element: this,
        setFormValue(v) {},
        setValidity(flags, message) {
           this.flags = flags || {};
           this.validationMessage = message || '';
           this.validity = { 
               valid: !Object.keys(flags || {}).length,
               customError: !!flags?.customError,
               ...flags
           };
        },
        validity: { valid: true },
        validationMessage: '',
        checkValidity() { return this.validity.valid; },
        reportValidity() { return this.validity.valid; }
      };
    }
    return this._internals;
  }
  attachShadow() {
    this.shadowRoot = new FakeShadowRoot(this.ownerDocument, this);
    return this.shadowRoot;
  }
  appendChild(node) {
    if (!node) return node;
    if (node.parentNode) {
      if (node.parentNode !== this) node.parentNode.removeChild?.(node);
      else {
        const existingIndex = this._childNodes.indexOf(node);
        if (existingIndex >= 0) this._childNodes.splice(existingIndex, 1);
      }
    }
    this._childNodes.push(node);
    if (typeof node === 'object') {
      node.parentNode = this;
      if (!node.ownerDocument && this.ownerDocument) node.ownerDocument = this.ownerDocument;
    }
    connectTree(node);
    return node;
  }
  insertBefore(newNode, referenceNode) {
    if (!referenceNode) return this.appendChild(newNode);
    const index = this._childNodes.indexOf(referenceNode);
    if (index === -1) return this.appendChild(newNode);
    if (newNode?.parentNode) {
      if (newNode.parentNode !== this) newNode.parentNode.removeChild?.(newNode);
      else {
        const existingIndex = this._childNodes.indexOf(newNode);
        if (existingIndex >= 0) this._childNodes.splice(existingIndex, 1);
      }
    }
    this._childNodes.splice(index, 0, newNode);
    if (typeof newNode === 'object') {
      newNode.parentNode = this;
      if (!newNode.ownerDocument && this.ownerDocument) newNode.ownerDocument = this.ownerDocument;
    }
    connectTree(newNode);
    return newNode;
  }
  removeChild(node) {
    const index = this._childNodes.indexOf(node);
    if (index >= 0) {
      this._childNodes.splice(index, 1);
      if (node && typeof node === 'object') node.parentNode = null;
    }
  }
  replaceWith(node) {
    if (!this.parentNode) return;
    const index = this.parentNode._childNodes.indexOf(this);
    if (index >= 0) {
      this.parentNode._childNodes.splice(index, 1, node);
      node.parentNode = this.parentNode;
    }
  }
  get childNodes() {
    return this._childNodes.filter(child => child?.parentNode === this);
  }
  remove() {
    if (this.parentNode) this.parentNode.removeChild(this);
  }
  get children() {
    return this.childNodes.filter(child => child?.nodeType === 1);
  }
  get firstElementChild() {
    return this.children[0] || null;
  }
  get nextElementSibling() {
    if (!this.parentNode) return null;
    const siblings = this.parentNode.children;
    const index = siblings.indexOf(this);
    return index >= 0 ? siblings[index + 1] || null : null;
  }
  get previousElementSibling() {
    if (!this.parentNode) return null;
    const siblings = this.parentNode.children;
    const index = siblings.indexOf(this);
    return index > 0 ? siblings[index - 1] || null : null;
  }
  setAttribute(name, value) {
    const stringValue = String(value);
    const prev = this.attributes.has(name) ? this.attributes.get(name) : null;
    this.attributes.set(name, stringValue);
    if (name === 'id') this.id = value;
    if (name === 'class') {
      this.classList = new FakeClassList();
      const classes = String(value).split(/\s+/).filter(Boolean);
      classes.forEach(cls => this.classList.add(cls));
    }
    if (name === 'value') {
      this._value = String(value ?? '');
    }
    if (name === 'checked') {
      this._checked = value !== false && value !== null && value !== 'false';
    }
    if (name === 'name') {
      this.name = String(value ?? '');
    }
    if (name === 'type') {
      this.type = String(value ?? '');
    }
    if (name === 'action') {
      this.action = String(value ?? '');
    }
    if (typeof this.attributeChangedCallback === 'function' && prev !== stringValue) {
      // console.log(`STUB: Attribute ${name} changed from ${prev} to ${stringValue} on ${this.tagName}`);
      this.attributeChangedCallback(name, prev, stringValue);
    }
  }
  getAttribute(name) {
    return this.attributes.has(name) ? this.attributes.get(name) : null;
  }
  hasAttribute(name) {
    return this.attributes.has(name);
  }
  removeAttribute(name) {
    const prev = this.attributes.has(name) ? this.attributes.get(name) : null;
    this.attributes.delete(name);
    if (name === 'id') this.id = undefined;
    if (name === 'class') this.classList = new FakeClassList();
    if (name === 'value') this._value = '';
    if (name === 'checked') this._checked = false;
    if (name === 'name') this._name = '';
    if (name === 'type') this._type = '';
    
    if (typeof this.attributeChangedCallback === 'function' && prev !== null) {
      this.attributeChangedCallback(name, prev, null);
    }
  }
  addEventListener(type, handler) {
    if (!this._listeners.has(type)) this._listeners.set(type, new Set());
    this._listeners.get(type).add(handler);
  }
  removeEventListener(type, handler) {
    this._listeners.get(type)?.delete(handler);
  }
  dispatchEvent(event) {
    const evt = typeof event === 'string' ? { type: event } : (event || {});
    evt.preventDefault ??= () => { evt.defaultPrevented = true; };
    evt.stopPropagation ??= () => { evt._stopped = true; };
    const handlers = Array.from(this._listeners.get(evt?.type) || []);
    try {
      if (evt && evt.target === undefined) evt.target = this;
      evt.currentTarget = this;
    } catch (err) {
      /* ignore read-only target */
    }
    handlers.forEach(handler => handler.call(this, evt));
    if (evt.bubbles && !evt._stopped) {
      if (this.parentNode?.dispatchEvent) {
        return this.parentNode.dispatchEvent(evt);
      }
      const win = this.ownerDocument?.defaultView || globalThis.window;
      if (win?.dispatchEvent) return win.dispatchEvent(evt);
    }
    return true;
  }
  contains(node) {
    return this === node || this.childNodes.some(child => child === node || (child.contains && child.contains(node)));
  }
  getRootNode(options) {
    let current = this;
    while (current.parentNode) {
      current = current.parentNode;
    }
    return current;
  }
  closest(selector) {
    if (!selector) return null;
    let current = this;
    while (current) {
      if (current.matches?.(selector)) return current;
      current = current.parentNode || current.host;
    }
    return null;
  }
  focus() {
    if (this.ownerDocument) this.ownerDocument.activeElement = this;
  }
  blur() {
    if (this.ownerDocument && this.ownerDocument.activeElement === this) {
      this.ownerDocument.activeElement = null;
    }
  }
  get innerHTML() {
    return this._innerHTML;
  }
  set innerHTML(value) {
    this._innerHTML = String(value ?? '');
    this._childNodes = [];
    this._textContent = '';
  }
  get textContent() {
    if (this.childNodes.length) {
      return this.childNodes.map(node => node.nodeType === 3 ? (node.textContent || '') : (node.textContent || '')).join('');
    }
    return this._textContent;
  }
  set textContent(value) {
    this._childNodes = [];
    this._textContent = String(value ?? '');
  }
  get value() {
    return this._value;
  }
  set value(val) {
    this._value = String(val ?? '');
    this.attributes.set('value', this._value);
  }
  get checked() {
    return !!this._checked;
  }
  set checked(val) {
    this._checked = !!val;
    if (this._checked) this.attributes.set('checked', '');
    else this.attributes.delete('checked');
  }
  get name() {
    return this._name;
  }
  set name(val) {
    this._name = String(val ?? '');
    this.attributes.set('name', this._name);
  }
  get type() {
    return this._type;
  }
  set type(val) {
    this._type = String(val ?? '');
    this.attributes.set('type', this._type);
  }
  get action() {
    return this._action || this.getAttribute('action') || '';
  }
  set action(val) {
    this._action = String(val ?? '');
    this.attributes.set('action', this._action);
  }
  getRootNode() {
    return this.ownerDocument || this;
  }
  getElementById(id) {
    return this.querySelector(`#${id}`);
  }
  querySelector(selector) {
    return this.querySelectorAll(selector)[0] || null;
  }
  querySelectorAll(selector) {
    const results = [];
    const walk = (node) => {
      if (!node || !node.childNodes) return;
      node.childNodes.forEach(child => {
        if (child?.matches?.(selector)) results.push(child);
        walk(child);
        if (child.shadowRoot) walk(child.shadowRoot);
      });
    };
    walk(this);
    return results;
  }
  matches(selector) {
    return matchSelector(this, selector);
  }
  submit() {
    this.dispatchEvent({
      type: 'submit',
      target: this,
      preventDefault() { this.defaultPrevented = true; }
    });
  }
  click() {
    if (this.type === 'checkbox') {
      this.checked = !this.checked;
      this.dispatchEvent({ type: 'click', target: this });
      this.dispatchEvent({ type: 'change', target: this });
      return;
    }
    this.dispatchEvent({ type: 'click', target: this });
  }
  reset() {
    const resetChild = (node) => {
      if (!node || !node.childNodes) return;
      node.childNodes.forEach(child => {
        if (child instanceof FakeElement) {
          if (child.tagName === 'INPUT' || child.tagName === 'TEXTAREA') {
            child.value = '';
            if (child.type === 'checkbox') child.checked = false;
          }
        }
        resetChild(child);
      });
    };
    resetChild(this);
  }
}

class FakeShadowRoot extends FakeElement {
  constructor(ownerDocument, host) {
    super('#shadow-root', ownerDocument);
    this.host = host;
    this.adoptedStyleSheets = [];
  }
}

class FakeHTMLElement extends FakeElement {
  constructor(tagName, ownerDocument) {
    super(tagName, ownerDocument);
  }
}

// --- MutationObserver Mock Support ---
let _mutationObservers = [];


class FakeMutationObserver {
  constructor(callback) {
    this.callback = callback;
    this._records = [];
    this._observed = [];
    this._active = false;
    _mutationObservers.push(this);
  }
  observe(target, options) {
    this._observed.push({ target, options: options || {} });
    this._active = true;
  }
  disconnect() {
    this._active = false;
    this._observed = [];
  }
  takeRecords() {
    const records = this._records.slice();
    this._records = [];
    return records;
  }
  _enqueue(record) {
    if (this._active) {
      this._records.push(record);
    }
  }
}

function isDescendantOrSelf(parent, node) {
  if (!parent || !node) return false;
  if (parent === node) return true;
  let current = node.parentNode;
  while (current) {
    if (current === parent) return true;
    current = current.parentNode;
  }
  return false;
}

function triggerMutation(target, record = null) {
  // record: { type, target, addedNodes, removedNodes, attributeName, oldValue, ... }
  _mutationObservers.forEach(observer => {
    if (!observer._active) return;
    for (const { target: observedTarget, options } of observer._observed) {
      if (observedTarget === target) {
        observer.callback([record || { type: 'childList', target, addedNodes: [], removedNodes: [] }], observer);
        return;
      }
      if (options && options.subtree && isDescendantOrSelf(observedTarget, target)) {
        observer.callback([record || { type: 'childList', target, addedNodes: [], removedNodes: [] }], observer);
        return;
      }
    }
  });
}

class FakeCustomEvent {
  constructor(type, options = {}) {
    this.type = type;
    this.detail = options.detail;
    this.bubbles = !!options.bubbles;
    this.composed = !!options.composed;
    this.target = null;
    this.defaultPrevented = false;
  }
  preventDefault() {
    this.defaultPrevented = true;
  }
  set currentTarget(value) {
    this.target = value;
  }
  get currentTarget() {
    return this.target;
  }
}

class FakeRange {
  constructor(context = null) {
    this.context = context;
    this.lastNode = null;
  }
  selectNodeContents(element) {
    this.context = element;
  }
  collapse() {
    /* no-op */
  }
  insertNode(node) {
    if (this.context) {
      this.context.appendChild(node);
      this.lastNode = node;
    }
  }
  setStartAfter(node) {
    this.lastNode = node;
  }
  deleteContents() {
    if (this.context) this.context._childNodes = [];
  }
  cloneRange() {
    const clone = new FakeRange(this.context);
    clone.lastNode = this.lastNode;
    return clone;
  }
}

class FakeSelection {
  constructor() {
    this._ranges = [];
    this.anchorNode = null;
  }
  get rangeCount() {
    return this._ranges.length;
  }
  getRangeAt(index) {
    return this._ranges[index];
  }
  removeAllRanges() {
    this._ranges = [];
    this.anchorNode = null;
  }
  addRange(range) {
    this._ranges = [range];
    this.anchorNode = range.context || range.lastNode || null;
  }
  deleteFromDocument() {
    const range = this._ranges[0];
    if (range?.context) range.context._childNodes = [];
  }
  collapseToEnd() {
    /* no-op */
  }
}

class FakeFormData {
  constructor(form) {
    this._entries = [];
    if (form) {
      const collect = (node) => {
        if (!node || !node.childNodes) return;
        node.childNodes.forEach(child => {
          if (child instanceof FakeElement) {
            const name = child.name || child.getAttribute('name');
            if (name) {
              this._entries.push([name, child.value]);
            }
          }
          collect(child);
        });
      };
      collect(form);
    }
  }
  get(name) {
    const entry = this._entries.find(([k]) => k === name);
    return entry ? entry[1] : null;
  }
  append(name, value) {
    this._entries.push([name, value]);
  }
  entries() {
    return this._entries[Symbol.iterator]();
  }
  [Symbol.iterator]() {
    return this.entries();
  }
}

class FakeDocument {
  constructor(selection, messageListeners) {
    this.activeElement = null;
    this._selection = selection;
    this._messageListeners = messageListeners;
    this._listeners = new Map();
    this.body = new FakeElement('body', this);
    this.documentElement = new FakeElement('html', this);
  }
  createElement(tag) {
    const name = String(tag || '');
    const registry = globalThis.customElements;
    const Ctor = registry?.get?.(name);
    let element;
    if (Ctor) {
      element = new Ctor();
      element.tagName = name.toUpperCase();
    } else {
      element = new FakeElement(name, this);
    }
    element.ownerDocument = this;
    return element;
  }
  createTextNode(text) {
    return {
      nodeType: 3,
      textContent: String(text ?? ''),
      remove() {
        this.parentNode?.removeChild?.(this);
      }
    };
  }
  createRange() {
    return new FakeRange();
  }
  querySelectorAll(selector) {
    return this.body.querySelectorAll(selector);
  }
  querySelector(selector) {
    return this.body.querySelector(selector);
  }
  execCommand() {
    /* no-op */
  }
  queryCommandSupported() {
    return false;
  }
  addEventListener(type, handler) {
    if (!this._listeners.has(type)) this._listeners.set(type, new Set());
    this._listeners.get(type).add(handler);
  }
  removeEventListener(type, handler) {
    this._listeners.get(type)?.delete(handler);
  }
  dispatchEvent(event) {
    const evt = typeof event === 'string' ? { type: event } : (event || {});
    evt.preventDefault ??= () => { evt.defaultPrevented = true; };
    evt.stopPropagation ??= () => { evt._stopped = true; };
    const handlers = this._listeners.get(evt?.type) || new Set();
    if (evt && evt.target === undefined) evt.target = this;
    evt.currentTarget = this;
    handlers.forEach(handler => handler.call(this, evt));
    if (evt.bubbles && !evt._stopped) {
      const win = this.defaultView || globalThis.window;
      if (win?.dispatchEvent) return win.dispatchEvent(evt);
    }
    return true;
  }
}

class FakeWindow {
  constructor(selection, messageListeners) {
    this._selection = selection;
    this._listeners = messageListeners || new Map();
    this.location = { origin: 'http://localhost', hostname: 'localhost' };
    this.document = null;
  }
  addEventListener(type, handler) {
    if (!this._listeners.has(type)) this._listeners.set(type, new Set());
    this._listeners.get(type).add(handler);
  }
  removeEventListener(type, handler) {
    this._listeners.get(type)?.delete(handler);
  }
  dispatchEvent(event) {
    const evt = typeof event === 'string' ? { type: event } : (event || {});
    evt.preventDefault ??= () => { evt.defaultPrevented = true; };
    evt.stopPropagation ??= () => { evt._stopped = true; };
    const handlers = this._listeners.get(evt?.type) || new Set();
    if (evt && evt.target === undefined) evt.target = this;
    evt.currentTarget = this;
    handlers.forEach(handler => handler.call(this, evt));
    return true;
  }
  postMessage(data) {
    const handlers = this._listeners.get('message') || new Set();
    handlers.forEach(handler => handler({ data }));
  }
  getSelection() {
    return this._selection;
  }
}

class FakeStorage {
  constructor() {
    this._store = new Map();
  }
  getItem(key) {
    return this._store.has(key) ? this._store.get(key) : null;
  }
  setItem(key, value) {
    this._store.set(key, String(value));
  }
  removeItem(key) {
    this._store.delete(key);
  }
  clear() {
    this._store.clear();
  }
}

const voidTags = new Set([
  'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input',
  'link', 'meta', 'param', 'source', 'track', 'wbr'
]);

const parseAttributes = (raw) => {
  const attrs = {};
  if (!raw) return attrs;
  const regex = /([^\s=]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+)))?/g;
  let match;
  while ((match = regex.exec(raw))) {
    const name = match[1];
    const value = match[2] ?? match[3] ?? match[4] ?? '';
    attrs[name] = value;
  }
  return attrs;
};

const extractBodyHtml = (html) => {
  const match = html.match(/<body[^>]*>([\s\S]*?)<\/body>/i);
  return match ? match[1] : html;
};

const findCallerFile = () => {
  const stack = new Error().stack?.split('\n').slice(2) || [];
  for (const line of stack) {
    if (line.includes('node:internal') || line.includes('node:fs')) continue;
    if (line.includes('pfusch-stubs.js')) continue;
    const urlMatch = line.match(/\((file:\/\/[^)]+)\)/) || line.match(/at (file:\/\/\S+)/);
    if (urlMatch) return fileURLToPath(urlMatch[1]);
    const pathMatch = line.match(/\((\/[^)]+):\d+:\d+\)/) || line.match(/at (\/\S+):\d+:\d+/);
    if (pathMatch) return pathMatch[1];
  }
  return null;
};

const PFUSCH_REMOTE_URL = new URL('https://matthiaskainer.github.io/pfusch/pfusch.js');
const PFUSCH_CDN_RE = /(['"])(?:https:\/\/matthiaskainer\.github\.io\/pfusch\/pfusch(?:\.min)?\.js|\.\/pfusch\.js)\1/g;
const pfuschImportCache = new Map();
const pfuschRemoteHash = createHash('sha1').update(PFUSCH_REMOTE_URL.href).digest('hex').slice(0, 8);
const pfuschRemotePath = path.join(os.tmpdir(), `pfusch.remote.${pfuschRemoteHash}.js`);
let pfuschRemotePromise = null;

const ensureRemotePfusch = async () => {
  if (pfuschRemotePromise) return pfuschRemotePromise;
  if (fs.existsSync(pfuschRemotePath)) return pathToFileURL(pfuschRemotePath);
  pfuschRemotePromise = (async () => {
    if (typeof original.fetch !== 'function') {
      throw new Error('import_for_test requires global fetch to download pfusch when no pfuschPath is provided');
    }
    const response = await original.fetch(PFUSCH_REMOTE_URL.href);
    if (!response.ok) {
      throw new Error(`import_for_test failed to download pfusch from ${PFUSCH_REMOTE_URL.href} (status ${response.status})`);
    }
    const source = await response.text();
    const tempPath = `${pfuschRemotePath}.tmp`;
    await fs.promises.writeFile(tempPath, source, 'utf8');
    await fs.promises.rename(tempPath, pfuschRemotePath);
    return pathToFileURL(pfuschRemotePath);
  })().catch(err => {
    pfuschRemotePromise = null;
    throw err;
  });
  return pfuschRemotePromise;
};

const resolveFileUrl = (specifier, parentFilePath) => {
  if (specifier instanceof URL) return specifier;
  if (typeof specifier !== 'string') {
    throw new Error('import_for_test expects a file path string or URL');
  }
  if (/^[a-zA-Z]+:/.test(specifier)) return new URL(specifier);
  if (path.isAbsolute(specifier)) return pathToFileURL(specifier);
  if (parentFilePath) {
    return pathToFileURL(path.resolve(path.dirname(parentFilePath), specifier));
  }
  return pathToFileURL(path.resolve(process.cwd(), specifier));
};

const resolveHtmlPath = (filePath) => {
  if (filePath instanceof URL) return fileURLToPath(filePath);
  if (typeof filePath !== 'string') {
    throw new Error('loadBaseDocument expects a file path string or URL');
  }
  if (path.isAbsolute(filePath)) return filePath;
  const caller = findCallerFile();
  if (caller) {
    const callerPath = path.resolve(path.dirname(caller), filePath);
    if (fs.existsSync(callerPath)) return callerPath;
  }
  const cwdPath = path.resolve(process.cwd(), filePath);
  if (fs.existsSync(cwdPath)) return cwdPath;
  return cwdPath;
};

const parseHtmlInto = (html, document, parent) => {
  const tokens = html.match(/<!--[\s\S]*?-->|<\/?[a-zA-Z][^>]*>|[^<]+/g) || [];
  const stack = [parent];
  tokens.forEach(token => {
    if (!token) return;
    if (token.startsWith('<!--')) return;
    if (token.startsWith('</')) {
      const tag = token.slice(2, -1).trim().toLowerCase();
      for (let i = stack.length - 1; i > 0; i -= 1) {
        if (stack[i]?.tagName?.toLowerCase() === tag) {
          stack.length = i;
          break;
        }
      }
      return;
    }
    if (token.startsWith('<')) {
      const selfClosing = token.endsWith('/>');
      const inner = token.slice(1, token.length - (selfClosing ? 2 : 1)).trim();
      if (!inner) return;
      const [tagName] = inner.split(/\s+/);
      const attrsRaw = inner.slice(tagName.length).trim();
      const el = document.createElement(tagName);
      const attrs = parseAttributes(attrsRaw);
      Object.entries(attrs).forEach(([name, value]) => {
        el.setAttribute(name, value);
      });
      stack[stack.length - 1].appendChild(el);
      if (!selfClosing && !voidTags.has(tagName.toLowerCase())) {
        stack.push(el);
      }
      return;
    }
    if (token.trim().length === 0) return;
    stack[stack.length - 1].appendChild(document.createTextNode(token));
  });
};

const refreshPfuschLightDom = (node) => {
  if (!node) return;
  if (node.lightDOMChildren !== undefined && node._lightById !== undefined) {
    node.lightDOMChildren = Array.from(node.children || []);
    const withIds = [
      ...node.lightDOMChildren,
      ...node.lightDOMChildren.flatMap(child => Array.from(child.querySelectorAll?.('[id]') || []))
    ].filter(child => child.id);
    node._lightById = new Map(withIds.map(child => [child.id, child]));
  }
  if (node.childNodes) node.childNodes.forEach(child => refreshPfuschLightDom(child));
};

class PfuschNodeCollection {
  constructor(nodes = [], host = null) {
    this.nodes = nodes.filter(Boolean);
    this.host = host;
  }
  _firstNode() {
    return this.nodes[0] || null;
  }
  get length() {
    return this.nodes.length;
  }
  get first() {
    const node = this._firstNode();
    return new PfuschNodeCollection(node ? [node] : [], this.host);
  }
  at(index) {
    return new PfuschNodeCollection([this.nodes[index]]);
  }
  get value() {
    return this._firstNode()?.value;
  }
  set value(val) {
    const node = this._firstNode();
    if (node) node.value = val;
  }
  get checked() {
    return !!this._firstNode()?.checked;
  }
  set checked(val) {
    const node = this._firstNode();
    if (node) node.checked = !!val;
  }
  get textContent() {
    return this._firstNode()?.textContent || '';
  }
  set textContent(val) {
    const node = this._firstNode();
    if (node) node.textContent = val;
  }
  submit() {
    this._firstNode()?.submit?.();
    return this;
  }
  click() {
    this._firstNode()?.click?.();
    return this;
  }
  get(selector) {
    const matches = [];
    this.nodes.forEach(node => {
      const scopes = [];
      if (node.shadowRoot) scopes.push(node.shadowRoot);
      scopes.push(node);
      scopes.forEach(scope => {
        if (scope?.querySelectorAll) {
          matches.push(...scope.querySelectorAll(selector));
        }
      });
    });
    return new PfuschNodeCollection(matches);
  }
  map(fn) {
    return this.nodes.map(node => fn(new PfuschNodeCollection([node], this.host)));
  }
  toArray() {
    return [...this.nodes];
  }
  get elements() {
    return this.nodes;
  }
  async flush() {
    await flushEffects();
    return this;
  }
}

export async function flushEffects() {
  await Promise.resolve();
  await Promise.resolve();
  await new Promise(resolve => setTimeout(resolve, 0));
}

export function pfuschTest(tagName, attributes = {}) {
  const element = document.createElement(tagName);
  Object.entries(attributes).forEach(([key, value]) => {
    const attrValue = typeof value === 'object' ? JSON.stringify(value) : value;
    element.setAttribute(key, attrValue);
  });
  document.body.appendChild(element);
  return new PfuschNodeCollection([element.shadowRoot || element], element);
}

export async function import_for_test(modulePath, pfuschPathOrOptions = null, extraReplacements = []) {
  const caller = findCallerFile();
  const moduleUrl = resolveFileUrl(modulePath, caller);
  if (moduleUrl.protocol !== 'file:') {
    throw new Error('import_for_test only supports local file paths for modules');
  }
  const moduleFsPath = fileURLToPath(moduleUrl);
  if (!fs.existsSync(moduleFsPath)) {
    throw new Error(`import_for_test could not find module at ${moduleFsPath}`);
  }

  let pfuschPath = pfuschPathOrOptions;
  let replacements = extraReplacements;

  if (pfuschPathOrOptions && typeof pfuschPathOrOptions === 'object') {
     pfuschPath = pfuschPathOrOptions.pfuschPath;
     replacements = pfuschPathOrOptions.replacements || replacements;
  }

  let pfuschUrl;
  if (pfuschPath == null) {
    pfuschUrl = await ensureRemotePfusch();
  } else {
    pfuschUrl = resolveFileUrl(pfuschPath, caller);
    if (pfuschUrl.protocol === 'file:' && !fs.existsSync(fileURLToPath(pfuschUrl))) {
      pfuschUrl = resolveFileUrl(pfuschPath, moduleFsPath);
    }
    if (pfuschUrl.protocol !== 'file:') {
      throw new Error('import_for_test only supports local file paths for pfusch');
    }
    const pfuschFsPath = fileURLToPath(pfuschUrl);
    if (!fs.existsSync(pfuschFsPath)) {
      throw new Error(`import_for_test could not find pfusch at ${pfuschFsPath}`);
    }
  }

  const cacheKey = `${moduleUrl.href}::${pfuschUrl.href}::${JSON.stringify(replacements)}`;
  const cached = pfuschImportCache.get(cacheKey);
  if (cached) return import(cached);

  const source = await fs.promises.readFile(moduleFsPath, 'utf8');
  let replaced = source.replace(PFUSCH_CDN_RE, `$1${pfuschUrl.href}$1`);

  if (replacements && replacements.length > 0) {
      for (const r of replacements) {
          replaced = replaced.replace(r.pattern, r.replacement);
      }
  }

  if (replaced === source) {
    if (source.match(PFUSCH_CDN_RE)) {
         throw new Error(`import_for_test regex failed to replace existing pfusch import in ${moduleFsPath}`);
    }
     // Else maybe it didn't have it (but we expect it usually)
  }

  const ext = path.extname(moduleFsPath) || '.js';
  const base = path.basename(moduleFsPath, ext);
  const dir = path.dirname(moduleFsPath);
  const hash = createHash('sha1').update(cacheKey).digest('hex').slice(0, 8);
  // Keep the shim next to the original so relative imports keep working.
  const tempPath = path.join(dir, `.${base}.pfusch-test.${hash}${ext}`);
  await fs.promises.writeFile(tempPath, replaced, 'utf8');

  const tempUrl = pathToFileURL(tempPath).href;
  const module = await import(tempUrl);
  pfuschImportCache.set(cacheKey, tempUrl);
  setTimeout(() => {
    fs.promises.unlink(tempPath).catch(() => {});
  }, 100);
  return module;
}

export async function loadBaseDocument(filePath) {
  const resolvedPath = resolveHtmlPath(filePath);
  const html = await fs.promises.readFile(resolvedPath, 'utf8');
  const bodyHtml = extractBodyHtml(html);
  const doc = globalThis.document;
  if (!doc?.body) {
    throw new Error('loadBaseDocument requires setupDomStubs() to be called first');
  }
  if (Array.isArray(doc.body._childNodes)) {
    doc.body._childNodes.forEach(child => {
      if (child?.parentNode === doc.body) child.parentNode = null;
    });
    doc.body._childNodes = [];
  }
  suspendConnect = true;
  try {
    parseHtmlInto(bodyHtml, doc, doc.body);
  } finally {
    suspendConnect = false;
  }
  refreshPfuschLightDom(doc.body);
  connectTree(doc.body);
  return new PfuschNodeCollection([doc.body]);
}

export async function loadDocumentFromString(html) {
  const bodyHtml = extractBodyHtml(html);
  const doc = globalThis.document;
  if (!doc?.body) {
    throw new Error('loadDocumentFromString requires setupDomStubs() to be called first');
  }
  if (Array.isArray(doc.body._childNodes)) {
    doc.body._childNodes.forEach(child => {
      if (child?.parentNode === doc.body) child.parentNode = null;
    });
    doc.body._childNodes = [];
  }
  suspendConnect = true;
  try {
    parseHtmlInto(bodyHtml, doc, doc.body);
  } finally {
    suspendConnect = false;
  }
  refreshPfuschLightDom(doc.body);
  connectTree(doc.body);
  return new PfuschNodeCollection([doc.body]);
}

export function setupDomStubs() {
  const messageListeners = new Map();
  const selection = new FakeSelection();
  const fakeWindow = new FakeWindow(selection, messageListeners);
  const fakeDocument = new FakeDocument(selection, messageListeners);
  function createSimpleFetchMock({ defaultPayload = { structuredContent: null } } = {}) {
    const calls = [];
    const routes = []; // [{ key, payload }], newest wins

    async function fetchMock(url, init) {
      const urlStr = String(url);
      calls.push({ url: urlStr, init, timestamp: Date.now() });

      const hit = routes.find((r) => urlStr.includes(r.key));
      const payload = hit ? hit.payload : defaultPayload;

      return {
        ok: true,
        json: async () => payload,
      };
    }

    fetchMock.addRoute = (key, payload) => {
      routes.unshift({ key, payload });
    };

    fetchMock.resetRoutes = () => {
      routes.length = 0;
    };

    fetchMock.getCalls = () => [...calls];

    fetchMock.resetCalls = () => {
      calls.length = 0;
    };

    return fetchMock;
  }
  globalThis.fetch = createSimpleFetchMock();
  globalThis.window = fakeWindow;
  globalThis.location = fakeWindow.location;
  globalThis.document = fakeDocument;
  fakeWindow.document = fakeDocument;
  fakeDocument.defaultView = fakeWindow;
  globalThis.Node = { TEXT_NODE: 3, ELEMENT_NODE: 1 };
  globalThis.CSSStyleSheet = class {
    replaceSync() { }
  };
  globalThis.HTMLElement = FakeElement;
  globalThis.KeyboardEvent = class KeyboardEvent {
        constructor(type, options) {
            this.type = type;
            this.key = options?.key || '';
            this.ctrlKey = options?.ctrlKey || false;
            this.metaKey = options?.metaKey || false;
            this.bubbles = options?.bubbles || false;
            this.cancelable = options?.cancelable || false;
            this._preventDefault = false;
        }
        preventDefault() { this._preventDefault = true; }
    };
  globalThis.matchMedia = () => ({
      matches: false,
      addListener: () => { },
      removeListener: () => { },
  });
  globalThis.localStorage = new FakeStorage();
  globalThis.sessionStorage = new FakeStorage();
  globalThis.customElements = {
    _registry: new Map(),
    define(name, ctor) {
      this._registry.set(name, ctor);
    },
    get(name) {
      return this._registry.get(name);
    }
  };
  globalThis.FormData = FakeFormData;
  globalThis.CustomEvent = FakeCustomEvent;
  globalThis.requestAnimationFrame = (cb) => setTimeout(cb, 0);
  globalThis.MutationObserver = FakeMutationObserver;
  globalThis.triggerMutation = triggerMutation;
  return {
    window: fakeWindow,
    document: fakeDocument,
    selection,
    restore() {
      globalThis.window = original.window || fakeWindow;
      globalThis.document = original.document || fakeDocument;
      globalThis.Node = original.Node || globalThis.Node;
      globalThis.CSSStyleSheet = original.CSSStyleSheet || globalThis.CSSStyleSheet;
      globalThis.customElements = original.customElements || globalThis.customElements;
      globalThis.HTMLElement = original.HTMLElement || globalThis.HTMLElement;
      globalThis.FormData = original.FormData || globalThis.FormData;
      globalThis.CustomEvent = original.CustomEvent || globalThis.CustomEvent;
      globalThis.requestAnimationFrame = original.requestAnimationFrame || globalThis.requestAnimationFrame;
      globalThis.fetch = original.fetch || globalThis.fetch;
      globalThis.KeyboardEvent = original.KeyboardEvent || globalThis.KeyboardEvent;
      globalThis.localStorage = original.localStorage || globalThis.localStorage;
      globalThis.sessionStorage = original.sessionStorage || globalThis.sessionStorage;
      globalThis.matchMedia = original.matchMedia || globalThis.matchMedia;
      globalThis.MutationObserver = original.MutationObserver || globalThis.MutationObserver;
      globalThis.triggerMutation = undefined;
    }
  };
}
