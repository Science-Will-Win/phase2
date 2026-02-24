# Inference UI - Visual Graph

A node-based graph editor for visually composing and running AI inference pipelines.

## Overview

Visual Graph is a node-based interface inspired by tools like Unity VFX Graph. Each step of an AI inference process is represented as a node, and nodes are connected together to visualize the execution plan.

<img src="img/graph-overview.png" alt="Visual Graph overview" width="800">

## Key Features

### Creating and Placing Nodes

Double-click or right-click on an empty area to open the Create Node menu. Nodes are organized by category. Select a node type to place it at the clicked position.

<img src="img/create-menu.png" alt="Create Node menu" width="480">

### Node Anatomy

Each node is composed of the following parts:

```
┌──────────────────────┐
│  ● Input Port (In)    │
├──────────────────────┤
│  [Step 1] Title       │  ← Header (drag handle)
│  Tool / Description   │  ← Body
│  ■■■■■■□□□ Progress   │  ← Progress bar
├──────────────────────┤
│  ● Output Port (Out)  │
└──────────────────────┘
```

- **Header**: Shows the step number and title. Double-click to edit the title. Drag to move.
- **Input/Output Ports**: Connection points to other nodes. Colors vary by port type.
- **Status Indicator**: A colored bar on the left side shows the execution state (pending, running, completed, error).

<img src="img/node-anatomy.png" alt="Node anatomy" width="480">

### Node Types

| Category | Nodes | Description |
|----------|-------|-------------|
| **General** | Step | Basic execution step |
| | Composite | Multi-step composite |
| | Visualize | Result visualization |
| | Observe | Intermediate result inspection |
| | Save | Result export |
| | Table | Tabular data display |
| **Input** | String | Text input |
| | Integer | Integer input |
| | Float | Float input |
| | Matrix3 / Matrix4 | Matrix input |
| **Data** | Data Loader | Load external data |
| | Image | Image data |
| **Tool** | Analyze | Analysis tool |
| | CodeGen | Code generation |
| | PubMed | PubMed search |
| | NCBI Gene | NCBI gene lookup |
| | CRISPR | CRISPR tool |
| | Protocol | Protocol generation |
| **Math** | Add, Subtract, Multiply, Divide, Power, Sqrt, Log | Arithmetic operations |

**Node Attributes:**

- `dataOnly` -- The node is a pure data supplier and is not included in the execution plan. It only provides values to other nodes via connections.
- `result` -- The node produces a visible result (display, save) rather than just passing data through.
- `allowRef` -- The node can receive reference connections. Only LLM-using nodes (Step, Composite, and all Tool nodes) have this flag. Nodes without it will reject reference connections and their ports will be dimmed during a ref drag.

#### General Nodes

<img src="img/nodes-general.png" alt="General category nodes" width="680">

| Node | Ports | Description |
|------|-------|-------------|
| **Step** | In(`any`) -> Out(`any`) | The fundamental execution unit. Configurable title, tool, and description. Shows a progress bar during execution. |
| **Composite** | Image(`image`) + Prompt(`string`) -> Out(`any`) | Combines an image and a text prompt for multimodal inference via a vision encoder. Image nodes must flow-connect through this node (direct Image-to-Step flow connections are not allowed). |
| **Visualize** | In(`any`) -> Out(`any`) | Displays images, charts, or visual results inline within the node body. `result`. |
| **Observe** | In(`any`) only | Terminal node that displays intermediate text results. Has no output port, so it cannot pass data further down the pipeline. `result`. |
| **Save** | In(`any`) only | Saves the incoming data to a file. Supports filename templates with dynamic tags such as `{date}`, `{time}`, `{uuid}`, `{node-title}`, etc. `result`. |
| **Table** | In(`table`) -> Out(`table`) | Renders incoming data as a table. Accepts `data` and `table` inputs. `result`. |

#### Input Nodes

<img src="img/nodes-input.png" alt="Input category nodes" width="680">

| Node | Ports | Description |
|------|-------|-------------|
| **String** | Out(`string`) | Provides a text value via an editable text area. `dataOnly`. |
| **Integer** | Out(`int`) | Provides an integer value via a number input. `dataOnly`. |
| **Float** | Out(`float`) | Provides a floating-point value via a number input. `dataOnly`. |
| **Double** | Out(`double`) | Provides a high-precision floating-point value. `dataOnly`. |
| **Boolean** | Out(`boolean`) | Provides a true/false value via a toggle checkbox. `dataOnly`. |
| **Vector2** | Out(`vector2`) | Provides a 2D vector as an editable 2-cell grid. `dataOnly`. |
| **Vector3** | Out(`vector3`) | Provides a 3D vector as an editable 3-cell grid. `dataOnly`. |
| **Vector4** | Out(`vector4`) | Provides a 4D vector as an editable 4-cell grid. `dataOnly`. |
| **Color** | Out(`color`) | Provides an RGBA color via a color picker and 4-channel number inputs. `dataOnly`. |
| **Matrix2** | Out(`matrix`) | Provides a 2x2 matrix as an editable grid. `dataOnly`. |
| **Matrix3** | Out(`matrix`) | Provides a 3x3 matrix as an editable grid. `dataOnly`. |
| **Matrix4** | Out(`matrix`) | Provides a 4x4 matrix as an editable grid. `dataOnly`. |

#### Data Nodes

| Node | Ports | Description |
|------|-------|-------------|
| **Data Loader** | Out(`data`) | Loads an external file via drag-and-drop or file picker. Shows filename and size. `dataOnly`. |
| **Image** | Out(`image`) | Loads an image file and shows a thumbnail preview. The `image` port type restricts flow connections to only `image`-compatible inputs (i.e. the Composite node). Reference connections to any node are still allowed. `dataOnly`. |

#### Tool Nodes

All Tool nodes share the same port layout: In(`any`) -> Out(`any`). Each node is pre-configured with a specific tool identifier.

| Node | Tool ID | Description |
|------|---------|-------------|
| **Analyze Plan** | `analyze_plan` | Analyzes execution plan results and produces a summary. |
| **Code Gen** | `code_gen` | Generates Python or R code based on the input context. |
| **PubMed Search** | `pubmed_search` | Searches PubMed for relevant research papers. |
| **NCBI Gene** | `ncbi_gene` | Queries the NCBI Gene database for gene information. |
| **CRISPR Designer** | `crispr_designer` | Designs CRISPR guide RNAs for gene editing experiments. |
| **Protocol Builder** | `protocol_builder` | Builds experimental protocols from research context. |

#### Math Nodes

<img src="img/nodes-math.png" alt="Math category nodes" width="680">

| Node | Ports | Description |
|------|-------|-------------|
| **Add** | A(`addable`) + B(`addable`) -> Out(dynamic) | Addition / string concatenation: A + B. Output type resolves dynamically -- `numeric` when all inputs are numeric, `string` when any input is string, `addable` by default. |
| **Subtract** | A(`numeric`) + B(`numeric`) -> Out(`numeric`) | Subtraction: A - B |
| **Multiply** | A(`numeric`) + B(`numeric`) -> Out(`numeric`) | Multiplication: A * B |
| **Divide** | A(`numeric`) + B(`numeric`) -> Out(`numeric`) | Division: A / B |
| **Power** | Base(`numeric`) + Exp(`numeric`) -> Out(`numeric`) | Exponentiation: Base ^ Exp |
| **Sqrt** | In(`numeric`) -> Out(`numeric`) | Square root |
| **Log** | In(`numeric`) -> Out(`numeric`) | Natural logarithm |

All Math nodes accept inline default values in their input fields when no connection is attached. The `numeric` group type accepts `float`, `int`, `double`, `matrix`, `vector2/3/4`, and `color` outputs. The `addable` group (used by Add) additionally accepts `string` for concatenation.

**Dynamic Output Types:** The Add node uses `resolveOutputType` to determine its output type based on connected inputs. When all inputs are `numeric` types, the output resolves to `numeric`. When any input is `string`, the output resolves to `string`. When no inputs are connected, the default output type is `addable`. Incompatible downstream connections are automatically removed when the output type changes.

#### Broadcasting Rules

Math nodes support broadcasting between different numeric types at runtime:

| Operation | scalar+scalar | string+string | string+numeric | matrix+scalar | matrix+matrix | vector+scalar | vector+vector | color+scalar | color+color |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Add** | Y | Y (concat) | Y (concat) | Y (elem) | Y (elem) | Y (elem) | Y (elem) | Y (elem) | Y (elem) |
| **Subtract** | Y | N | N | Y | Y | Y | Y | Y | Y |
| **Multiply** | Y | N | N | Y (scalar) | Y (elem) | Y (scalar) | Y (elem) | Y (scalar) | Y (elem) |
| **Divide** | Y | N | N | Y | Y | Y | Y | Y | Y |
| **Power** | Y | N | N | Y | Y | Y | Y | Y | Y |
| **Sqrt** | Y | N | - | Y | - | Y | - | Y | - |
| **Log** | Y | N | - | Y | - | Y | - | Y | - |

("scalar" = float, int, double)

### Connections

Drag from an output port to an input port to create a connection.

<img src="img/connection-drag.png" alt="Creating a connection by dragging" width="480">

**Connection Types:**

| Type | Mouse Action | Line Style | Purpose |
|------|-------------|------------|---------|
| Flow | Left-click drag | Solid | Execution order and data flow |
| Reference | Right-click drag | Dashed | Reference data attachment |

- When dragging near a compatible port (within 15px), the cursor automatically snaps to it.
- Dragging from an input port that already has a connection detaches it so you can reconnect.
- Right-click a connection line to delete it.

**Flow Connections** represent the primary execution order and data pipeline. Data produced by one node's output flows into the next node's input along these solid lines. The graph engine executes nodes following the flow connection order.

**Reference Connections** are a secondary connection type used to attach supplementary data to a node without being part of the main execution chain. For example, a Step node might reference a Data Loader node to access a dataset during its execution, even though the Data Loader is not the previous step in the pipeline. Reference connections are shown as dashed lines to visually distinguish them from the primary flow.

To create a reference connection, right-click drag from a port instead of left-click dragging. Unlike flow connections, reference connections bypass port type restrictions. However, only nodes with `allowRef: true` can receive reference connections -- these are the LLM-using nodes: Step, Composite, and all Tool nodes (Analyze, CodeGen, PubMed, NCBI Gene, CRISPR, Protocol). Input, Data, Math, and result nodes do not accept reference connections and their ports will be dimmed during a ref drag. This makes it possible to attach an Image node to a Step node via reference even though a direct flow connection is not allowed.

<img src="img/reference-connection.png" alt="Reference connection example" width="480">

**Port Type Compatibility (Flow Connections):**

During a flow connection drag, compatible ports are highlighted and incompatible ones are dimmed. The type system uses a `PortTypes` registry where types declare which groups they belong to. This enables extensibility -- adding a new type (e.g. `quaternion`) only requires a single `PortTypes.define()` call, and existing group-typed ports automatically accept it.

- `any` is compatible with all port types except `image`
- `image` can only flow-connect to `image` inputs (e.g. the Composite node's Image port)
- `numeric` (group) accepts `float`, `int`, `double`, `matrix`, `vector2/3/4`, `color`
- `addable` (group) accepts everything `numeric` does, plus `string`
- `string` inputs accept any output type
- `float` inputs also accept `int` outputs
- `table` inputs accept `table` and `data` outputs
- `boolean`, `data`, `table` are standalone types (not in any group)
- Reference connections bypass type restrictions but only on `allowRef: true` nodes (Step, Composite, Tool nodes)

<img src="img/port-compatibility.png" alt="Port compatibility highlighting" width="680">

### Selection and Editing

**Single Selection:**
- Click a node to select it. Selected nodes show a blue border with a glow effect.

**Multi-Selection:**
- Drag on an empty area to create a marquee (rectangle) selection
- Ctrl/Cmd+Click to toggle individual nodes in the selection
- Drag any selected node to move all selected nodes together

<img src="img/multi-select.png" alt="Multi-selection with marquee" width="680">

**Keyboard Shortcuts:**

| Shortcut | Action |
|----------|--------|
| Delete / Backspace | Delete selected nodes or connections |
| Ctrl+Z | Undo (up to 50 actions) |
| Ctrl+Shift+Z | Redo |
| Ctrl+C | Copy selected nodes |
| Ctrl+V | Paste |
| Double-click header | Edit node title |

### View Controls

| Action | Method |
|--------|--------|
| Pan | Right-click drag on empty area |
| Zoom | Mouse wheel |
| Zoom In | Zoom control + button |
| Zoom Out | Zoom control - button |
| Reset Zoom | Zoom control reset button |

### State Persistence

The graph state (node positions, connections, pan/zoom) is automatically saved to `localStorage` whenever nodes are placed, moved, or connections change. The state is restored when the page reloads or when switching between conversations.

## Architecture

```
┌─────────────────────────────────────────────┐
│  index.html                                  │
│  ┌─────────────┐  ┌──────────────────────┐  │
│  │ node-       │  │ NodeRegistry         │  │
│  │ registry.js │──│ .register()          │  │
│  │             │  │ .get() / .getAll()   │  │
│  └─────────────┘  └──────────────────────┘  │
│        ↓ loadAll()                           │
│  ┌─────────────────────────────────────┐    │
│  │ nodes/*/*.js  (auto-loaded)         │    │
│  │ step, string, math_add, ...         │    │
│  └─────────────────────────────────────┘    │
│        ↓                                     │
│  ┌─────────────┐      ┌───────────────┐     │
│  │ node-       │      │ node-         │     │
│  │ graph.js    │      │ graph.css     │     │
│  │ (engine)    │      │ (styles)      │     │
│  └─────────────┘      └───────────────┘     │
│        ↓                                     │
│  ┌─────────────┐                             │
│  │ app.js      │  ← App integration,        │
│  │             │    save/restore state       │
│  └─────────────┘                             │
└─────────────────────────────────────────────┘
```

### Core Files

| File | Role |
|------|------|
| `node-graph.js` | Graph engine (rendering, connections, selection, drag, undo/redo) |
| `node-graph.css` | Visual styles for nodes, ports, connections, menus |
| `nodes/node-registry.js` | Node type registry |
| `nodes/*/*.js` | Individual node type definitions |
| `app.js` | App integration (localStorage save/restore, execution plan sync) |
| `index.html` | Main page (contains graph container) |
| `graph.html` | Standalone graph page for pop-out windows |

## Required Images

Please prepare the following images in the `inference_ui/img/` folder:

| Filename | Content |
|----------|---------|
| `graph-overview.png` | Full screenshot of the graph with several nodes connected together |
| `create-menu.png` | Screenshot showing the Create Node menu open |
| `node-anatomy.png` | Zoomed-in view of a single node with labels on each part (header, ports, body, status bar) |
| `connection-drag.png` | Screenshot of a connection being dragged from one port to another |
| `port-compatibility.png` | Screenshot showing compatible ports highlighted and incompatible ports dimmed during a drag |
| `multi-select.png` | Screenshot of multiple nodes selected via marquee drag (showing blue border highlights) |
| `reference-connection.png` | Screenshot showing a dashed reference connection between two nodes alongside a solid flow connection, so both types are visible for comparison |
| `nodes-general.png` | All General category nodes side by side: Step, Composite, Visualize, Observe, Save, Table |
| `nodes-input.png` | All Input category nodes side by side: String, Integer, Float, Double, Boolean, Vector2, Vector3, Vector4, Color, Matrix2, Matrix3, Matrix4 |
| `nodes-math.png` | All 7 Math category nodes side by side: Add, Subtract, Multiply, Divide, Power, Sqrt, Log, showing the inline default value inputs |

## Adding Custom Nodes

To add new node types, see [ADDING_NODES.md](ADDING_NODES.md).
