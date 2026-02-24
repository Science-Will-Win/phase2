# Custom Node Guide

This document explains how to add new node types to the Visual Graph system.

## Directory Structure

```
inference_ui/
├── nodes/
│   ├── node-registry.js          # PortTypes + NodeRegistry (global)
│   ├── step/step-node.js         # General: Step
│   ├── string/string-node.js     # Input: String
│   ├── integer/integer-node.js   # Input: Integer
│   ├── float/float-node.js       # Input: Float
│   ├── double/double-node.js     # Input: Double
│   ├── boolean/boolean-node.js   # Input: Boolean
│   ├── vector2/vector2-node.js   # Input: Vector2
│   ├── vector3/vector3-node.js   # Input: Vector3
│   ├── vector4/vector4-node.js   # Input: Vector4
│   ├── color/color-node.js       # Input: Color
│   ├── matrix2/matrix2-node.js   # Input: Matrix 2x2
│   ├── matrix3/matrix3-node.js   # Input: Matrix 3x3
│   ├── matrix4/matrix4-node.js   # Input: Matrix 4x4
│   ├── math/
│   │   ├── math-helper.js        # Shared renderer for Math nodes
│   │   ├── add-node.js           # Math: Add
│   │   ├── subtract-node.js      # Math: Subtract
│   │   └── ...
│   ├── data/data-node.js         # Data: Data Loader
│   ├── image/image-node.js       # Data: Image
│   ├── visualize/visualize-node.js
│   ├── observe/observe-node.js
│   ├── save/save-node.js
│   ├── table/table-node.js
│   ├── analyze/analyze-node.js   # Tool: Analyze
│   ├── pubmed/pubmed-node.js     # Tool: PubMed
│   └── ...
├── node-graph.js                 # Graph engine
├── node-graph.css                # Graph styles
└── app.js                        # App integration
```

## Quick Start: Add a Node in 3 Steps

### 1. Create a Folder and File

```
nodes/
└── my-custom/
    └── my-custom-node.js
```

### 2. Write the Node Definition

```javascript
// nodes/my-custom/my-custom-node.js

NodeRegistry.register('my_custom', {
    label: 'My Custom Node',
    category: 'Tool',

    ports: [
        { name: 'in',  dir: 'in',  type: 'string', label: 'Input' },
        { name: 'out', dir: 'out', type: 'string', label: 'Output' }
    ],

    defaultConfig: {
        title: 'My Custom',
        status: 'pending'
    },

    render(node, helpers) {
        const el = document.createElement('div');
        el.innerHTML = `
            <div class="ng-ports-in-row" data-node-id="${node.id}">
                <div class="ng-port ng-port-in"
                     data-port-name="in" data-port-dir="in"
                     data-port-type="string" data-node-id="${node.id}">
                </div>
            </div>
            <div class="ng-node-header">
                <span class="ng-node-title">${helpers.escapeHtml(node.title)}</span>
            </div>
            <div class="ng-node-body">Custom content here</div>
            <div class="ng-ports-out-row" data-node-id="${node.id}">
                <div class="ng-port ng-port-out"
                     data-port-name="out" data-port-dir="out"
                     data-port-type="string" data-node-id="${node.id}">
                </div>
            </div>
        `;
        return el;
    },

    getDragHandle(el) {
        return el.querySelector('.ng-node-header');
    }
});
```

### 3. Restart the Server

Node files are discovered automatically via the `/api/node-manifest` endpoint. Restarting the server picks up any new files under `nodes/`.

---

## API Reference

### NodeRegistry.register(type, definition)

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `type` | string | Yes | Unique type identifier (e.g. `'my_custom'`) |
| `definition` | object | Yes | Node definition object |

### Definition Object

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| `label` | string | No | Display name in the menu (defaults to type) |
| `category` | string | No | Category group (defaults to `'General'`) |
| `ports` | Port[] | No | Array of port definitions |
| `defaultConfig` | object | No | Default values when a node is created |
| `render(node, helpers)` | function | Yes | Returns a DOM element for the node |
| `afterRender(el, node, helpers)` | function | No | Post-render setup (event listeners, etc.) |
| `getDragHandle(el)` | function | No | Returns the drag handle element |
| `updateStatus(el, status)` | function | No | Custom UI update on status change |
| `updateResult(el, resultText)` | function | No | Update result text display |
| `dataOnly` | boolean | No | Marks a data-only node |
| `result` | boolean | No | Marks a node that produces a visible result (display, save) |
| `resolveOutputType` | function | No | Dynamic output type resolver. Receives an array of connected input types and returns the resolved output type string. See the Add node for an example. |
| `allowRef` | boolean | No | Allows the node to receive reference connections. Only LLM-using nodes (General: Step/Composite, all Tool nodes) should set this to `true`. Nodes without `allowRef: true` will have their ports dimmed and will reject ref connections. |

### Port Definition

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| `name` | string | Yes | Unique port name within the node |
| `dir` | `'in'` \| `'out'` | Yes | Direction (input or output) |
| `type` | string | Yes | Data type |
| `label` | string | No | Display label |
| `defaultValue` | any | No | Default value (input ports only) |

### Port Types and Compatibility

The system uses a `PortTypes` registry (defined in `node-registry.js`) that supports **concrete types** and **group types**.

#### Concrete Types

| Type | Description | Flow Connect To (output) |
|------|-------------|--------------------------|
| `any` | Universal | All ports except `image` |
| `string` | Text | All ports (string inputs accept anything) |
| `int` | Integer | `int`, `float`, `numeric`, `addable`, `any`, `string` |
| `float` | Float | `float`, `numeric`, `addable`, `any`, `string` |
| `double` | High-precision float | `double`, `numeric`, `addable`, `any`, `string` |
| `boolean` | True/false | `boolean`, `any`, `string` |
| `matrix` | Matrix array | `matrix`, `numeric`, `addable`, `any`, `string` |
| `vector2` | 2D vector | `vector2`, `numeric`, `addable`, `any`, `string` |
| `vector3` | 3D vector | `vector3`, `numeric`, `addable`, `any`, `string` |
| `vector4` | 4D vector | `vector4`, `numeric`, `addable`, `any`, `string` |
| `color` | RGBA color | `color`, `numeric`, `addable`, `any`, `string` |
| `data` | File/data object | `data`, `table`, `any`, `string` |
| `table` | Tabular data | `table`, `any`, `string` |
| `image` | Image data | `image` only |

#### Group Types (used as input port types)

| Group | Accepts | Used By |
|-------|---------|---------|
| `numeric` | `float`, `int`, `double`, `matrix`, `vector2/3/4`, `color` | Multiply, Subtract, Divide, Power, Sqrt, Log inputs/outputs |
| `addable` | Everything in `numeric` + `string` | Add inputs |

A group-typed output (e.g. `numeric`) is compatible with another group input if it is a subset (e.g. `numeric` -> `addable` works because all `numeric` members are in `addable`).

#### Compatibility Rules

- `image` is isolated: it can only flow-connect to `image` inputs (e.g. Composite's Image port)
- `any` is compatible with everything except `image`
- `table` inputs accept `table` and `data` outputs
- Group inputs (`numeric`, `addable`) accept any registered member type
- `string` input ports accept any output type (coercion)
- `float` input ports also accept `int` outputs (promotion)
- Otherwise, types must match exactly

**Reference connections** bypass port type restrictions, but are only accepted by nodes with `allowRef: true` (Step, Composite, and all Tool nodes). Nodes without this property (Input, Data, Math, result nodes) will reject reference connections. Allowed nodes are additionally subject to the max attachments limit.

### Registering Custom Port Types

If your node introduces a new data type, register it with `PortTypes` at the top of your node file (before `NodeRegistry.register()`):

```javascript
// Register a new type that works with Math nodes
PortTypes.define('quaternion', ['numeric', 'addable']);

NodeRegistry.register('quaternion_value', {
    label: 'Quaternion',
    category: 'Input',
    ports: [{ name: 'out', dir: 'out', type: 'quaternion', label: 'Value' }],
    // ...
});
```

The second argument is an array of group names this type belongs to. After registration, all `numeric` and `addable` input ports automatically accept `quaternion` outputs. To create a standalone type (not in any group), pass an empty array:

```javascript
PortTypes.define('audio', []);
```

### Dynamic Output Types (resolveOutputType)

If a node's output type depends on what is connected to its inputs, define a `resolveOutputType(inputTypes)` function in the node definition. The graph engine calls this whenever input connections change and updates the output port type, color, and connection line colors automatically.

```javascript
NodeRegistry.register('my_polymorphic', {
    label: 'Polymorphic',
    category: 'Math',

    ports: [
        { name: 'a', dir: 'in', type: 'addable', label: 'A', defaultValue: 0 },
        { name: 'out', dir: 'out', type: 'addable', label: 'Result' }
    ],

    resolveOutputType(inputTypes) {
        if (inputTypes.length === 0) return 'addable';
        if (inputTypes.some(t => t === 'string')) return 'string';
        return 'numeric';
    },

    // ...
});
```

When the resolved output type changes, any downstream flow connections that become incompatible with the new type are automatically removed.

### Helpers in render()

| Helper | Description |
|--------|-------------|
| `helpers.escapeHtml(str)` | Escapes HTML special characters |

### Extended Helpers in afterRender()

| Helper | Description |
|--------|-------------|
| `helpers.escapeHtml(str)` | Escapes HTML special characters |
| `helpers.getNode()` | Returns the current node data object |
| `helpers.rerender(node)` | Re-renders the node from scratch |
| `helpers.updateConnections(node)` | Recalculates connection line positions |

---

## Categories

Currently used categories:

- **General** — Step, Composite, Visualize, Observe, Save, Table
- **Input** — String, Integer, Float, Double, Boolean, Vector2, Vector3, Vector4, Color, Matrix2, Matrix3, Matrix4
- **Data** — Data Loader, Image
- **Tool** — Analyze, CodeGen, CRISPR, PubMed, NCBI Gene, Protocol
- **Math** — Add, Subtract, Multiply, Divide, Power, Sqrt, Log

Using a new category name in `category` will automatically add it to the Create Node menu.

---

## Connection Types

Node connections come in two types:

| Type | How to Create | Visual Style | Purpose |
|------|--------------|--------------|---------|
| `flow` | Left-click drag from a port | Solid line | Execution order / data flow |
| `ref` | Right-click drag from a port | Dashed line | Reference attachment |

Flow connections enforce port type compatibility (see above). Reference connections bypass port type restrictions but are only accepted by nodes with `allowRef: true` (Step, Composite, and all Tool nodes). For example, an Image node (`image` output) cannot flow-connect to a Step node (`any` input), but it can be attached as a reference because Step has `allowRef: true`. Math and Input nodes do not accept reference connections.

---

## Practical Examples

### Node with Default Value Input Fields

```javascript
NodeRegistry.register('threshold', {
    label: 'Threshold',
    category: 'Tool',

    ports: [
        { name: 'data_in', dir: 'in', type: 'data', label: 'Data' },
        { name: 'value',   dir: 'in', type: 'float', label: 'Threshold', defaultValue: 0.5 },
        { name: 'out',     dir: 'out', type: 'data', label: 'Filtered' }
    ],

    defaultConfig: {
        title: 'Threshold Filter',
        status: 'pending',
        portValues: { value: 0.5 }
    },

    render(node, helpers) {
        const el = document.createElement('div');
        const val = node.portValues?.value ?? 0.5;

        el.innerHTML = `
            <div class="ng-ports-in-row" data-node-id="${node.id}">
                <div class="ng-port ng-port-in"
                     data-port-name="data_in" data-port-dir="in"
                     data-port-type="data" data-node-id="${node.id}">
                </div>
                <span class="ng-port-label">Data</span>
            </div>
            <div class="ng-node-header">
                <span class="ng-node-title">${helpers.escapeHtml(node.title)}</span>
            </div>
            <div class="ng-node-body">
                <div class="ng-port-field" data-port-ref="value" style="display:flex;align-items:center;gap:6px;">
                    <div class="ng-port ng-port-in"
                         data-port-name="value" data-port-dir="in"
                         data-port-type="float" data-node-id="${node.id}">
                    </div>
                    <span class="ng-port-label">Threshold</span>
                    <input type="number" class="ng-port-default ng-interactive"
                           value="${val}" step="0.1" data-port-ref="value">
                </div>
            </div>
            <div class="ng-ports-out-row" data-node-id="${node.id}">
                <div class="ng-port ng-port-out"
                     data-port-name="out" data-port-dir="out"
                     data-port-type="data" data-node-id="${node.id}">
                </div>
            </div>
        `;
        return el;
    },

    getDragHandle(el) {
        return el.querySelector('.ng-node-header');
    }
});
```

### Output-Only Data Source Node

```javascript
NodeRegistry.register('constant', {
    label: 'Constant',
    category: 'Input',
    dataOnly: true,

    ports: [
        { name: 'out', dir: 'out', type: 'float', label: 'Value' }
    ],

    defaultConfig: {
        title: 'Constant',
        status: 'completed',
        portValues: { out: 0 }
    },

    render(node, helpers) {
        const el = document.createElement('div');
        const val = node.portValues?.out ?? 0;
        el.innerHTML = `
            <div class="ng-node-header">
                <span class="ng-node-title">${helpers.escapeHtml(node.title)}</span>
            </div>
            <div class="ng-input-node-body">
                <input type="number" class="ng-input-node-field ng-port-default ng-interactive"
                       value="${val}" step="any" data-port-ref="out">
            </div>
            <div class="ng-ports-out-row" data-node-id="${node.id}">
                <div class="ng-port ng-port-out"
                     data-port-name="out" data-port-dir="out"
                     data-port-type="float" data-node-id="${node.id}">
                </div>
            </div>
        `;
        return el;
    },

    getDragHandle(el) {
        return el.querySelector('.ng-node-header');
    }
});
```

---

## Checklist

Verify the following when adding a new node:

- [ ] Created file at `nodes/<folder>/<name>-node.js`
- [ ] Used a unique type string in `NodeRegistry.register()`
- [ ] `render()` returns a DOM element
- [ ] Port elements have correct `data-port-name`, `data-port-dir`, `data-port-type`, and `data-node-id` attributes
- [ ] `getDragHandle()` defined (falls back to `.ng-node-header` if omitted)
- [ ] Interactive input elements have the `ng-interactive` class (prevents drag conflicts)
- [ ] Input elements for default values have the `ng-port-default` class (auto-wires change events)
- [ ] Confirmed the node appears in the Create Node menu after server restart
