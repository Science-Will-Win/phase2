// ============================================
// Node Graph Engine - Unity VFX Style
// ============================================

class NodeGraph {
    constructor(container) {
        this.container = container;
        this.svg = container.querySelector('.node-graph-svg');
        this.canvas = container.querySelector('.node-graph-canvas');
        if (!this.svg || !this.canvas) return;

        this.nodes = new Map();
        this.connections = new Map();
        this.nextNodeId = 1;
        this.nextConnId = 1;

        // Pan/zoom state
        this.panX = 0;
        this.panY = 0;
        this.scale = 1;
        this.minScale = 0.2;
        this.maxScale = 3;
        this.isPanning = false;
        this.panStartX = 0;
        this.panStartY = 0;

        // Node drag state
        this.dragNode = null;
        this.dragOffsetX = 0;
        this.dragOffsetY = 0;

        // Node resize state
        this.resizeState = null;

        // Deferred layout flag (set when layout is computed while graph tab is hidden)
        this._needsLayout = null;

        // Connection drag state
        this.pendingConn = null;

        // Selection
        this.selectedConnection = null;
        this.selectedNode = null;
        this.selectedNodes = new Set();

        // Marquee / clipboard state
        this.marqueeState = null;
        this.clipboard = null;

        // Group drag state
        this._groupDragStart = null;

        // Pending connection undo tracking
        this._undoPushedForPendingConn = false;

        // Undo/Redo history
        this._undoStack = [];
        this._redoStack = [];
        this._maxHistory = 50;

        // Context menu (Create Node)
        this._contextMenu = null;
        this._contextMenuGraphPos = { x: 0, y: 0 };
        this._rightClickStart = null;
        this._panBeforeRightClick = null;

        // Execution plan diff tracking
        this._lastExecutionPlanHash = null;

        this._setupEvents();
        this._applyTransform();
    }

    // ---- Port Helpers ----

    _getPortsForNode(nodeId) {
        const node = this.nodes.get(nodeId);
        if (!node) return [];
        const nodeDef = NodeRegistry.get(node.type);
        let ports = nodeDef?.ports || [
            { name: 'in', dir: 'in', type: 'any' },
            { name: 'out', dir: 'out', type: 'any' }
        ];
        if (node._resolvedOutType) {
            ports = ports.map(p =>
                p.dir === 'out' && p.name === 'out'
                    ? { ...p, type: node._resolvedOutType }
                    : p
            );
        }
        return ports;
    }

    _isTypeCompatible(outType, inType) {
        return PortTypes.isCompatible(outType, inType);
    }

    _resolveOutputTypes(nodeId, visited) {
        if (!visited) visited = new Set();
        if (visited.has(nodeId)) return;
        visited.add(nodeId);

        const node = this.nodes.get(nodeId);
        if (!node) return;
        const def = NodeRegistry.get(node.type);
        if (!def?.resolveOutputType) return;

        const inputTypes = [];
        for (const conn of this.connections.values()) {
            if (conn.to === nodeId && conn.type === 'flow') {
                const fromPorts = this._getPortsForNode(conn.from);
                const fromPortDef = fromPorts.find(p => p.name === conn.fromPort);
                if (fromPortDef) inputTypes.push(fromPortDef.type);
            }
        }

        const newOutType = def.resolveOutputType(inputTypes);
        const oldOutType = node._resolvedOutType;
        node._resolvedOutType = newOutType;

        const outPortEl = this.canvas.querySelector(
            `.ng-port-out[data-node-id="${nodeId}"][data-port-name="out"]`
        );
        if (outPortEl) outPortEl.setAttribute('data-port-type', newOutType);

        if (oldOutType !== newOutType) {
            const toRemove = [];
            for (const [connId, conn] of this.connections) {
                if (conn.from !== nodeId || conn.type !== 'flow') continue;
                const toPorts = this._getPortsForNode(conn.to);
                const toPortDef = toPorts.find(p => p.name === conn.toPort);
                if (toPortDef && !this._isTypeCompatible(newOutType, toPortDef.type)) {
                    toRemove.push(connId);
                }
            }
            for (const connId of toRemove) this.removeConnection(connId);

            for (const conn of this.connections.values()) {
                if (conn.from === nodeId) {
                    const visPaths = this.svg.querySelectorAll(
                        `.ng-connection[data-conn-id="${conn.id}"]`
                    );
                    visPaths.forEach(vp => {
                        vp.className.baseVal.split(' ')
                            .filter(c => c.startsWith('ng-conn-type-'))
                            .forEach(c => vp.classList.remove(c));
                        if (newOutType !== 'any') {
                            vp.classList.add(`ng-conn-type-${newOutType}`);
                        }
                    });
                }
            }

            for (const conn of this.connections.values()) {
                if (conn.from === nodeId && conn.type === 'flow') {
                    this._resolveOutputTypes(conn.to, visited);
                }
            }
        }
    }

    _resolveAllOutputTypes() {
        for (const nodeId of this.nodes.keys()) {
            this._resolveOutputTypes(nodeId);
        }
    }

    _allowsRef(nodeId) {
        const node = this.nodes.get(nodeId);
        if (!node) return false;
        const def = NodeRegistry.get(node.type);
        return def?.allowRef === true;
    }

    // ---- Event Setup ----

    _setupEvents() {
        this.container.addEventListener('contextmenu', e => e.preventDefault());

        this.container.addEventListener('mousedown', e => {
            if (this.pendingConn) {
                if (e.button === 0 && this.pendingConn.connType === 'ref') {
                    this.pendingConn.connType = 'flow';
                    this.pendingConn.tempLine.classList.replace('ng-conn-pending-ref', 'ng-conn-pending');
                    this._clearPortHighlights();
                    this._highlightCompatiblePorts();
                } else if (e.button === 2 && this.pendingConn.connType === 'flow') {
                    this.pendingConn.connType = 'ref';
                    this.pendingConn.tempLine.classList.replace('ng-conn-pending', 'ng-conn-pending-ref');
                    this._clearPortHighlights();
                    this._highlightCompatiblePorts();
                }
                e.preventDefault();
                return;
            }
            if (e.button === 2) {
                this._rightClickStart = { x: e.clientX, y: e.clientY };
                this._panBeforeRightClick = { x: this.panX, y: this.panY };
                this.isPanning = true;
                this.panStartX = e.clientX - this.panX;
                this.panStartY = e.clientY - this.panY;
                this.container.style.cursor = 'grabbing';
                e.preventDefault();
            }
            if (e.button === 0 && this._contextMenu) {
                if (!this._contextMenu.contains(e.target)) {
                    this._closeCreateMenu();
                }
            }
            if (e.button === 0 && this._isBackground(e.target)) {
                const append = e.ctrlKey || e.metaKey || e.shiftKey;
                this._startMarquee(e, append);
                e.preventDefault();
                return;
            }
            if (this._isBackground(e.target)) {
                this._clearSelection();
            }
        });

        window.addEventListener('mousemove', e => {
            if (this.isPanning) {
                this.panX = e.clientX - this.panStartX;
                this.panY = e.clientY - this.panStartY;
                this._applyTransform();
                return;
            }
            if (this.resizeState) {
                this._handleResize(e);
                return;
            }
            if (this.marqueeState) {
                this._updateMarquee(e);
                return;
            }
            if (this.dragNode) {
                const rect = this.container.getBoundingClientRect();
                const x = (e.clientX - rect.left - this.panX) / this.scale - this.dragOffsetX;
                const y = (e.clientY - rect.top - this.panY) / this.scale - this.dragOffsetY;
                if (this._groupDragStart && this._groupDragStart.has(this.dragNode)) {
                    const anchorStart = this._groupDragStart.get(this.dragNode);
                    const dx = x - anchorStart.x;
                    const dy = y - anchorStart.y;
                    for (const [nid, startPos] of this._groupDragStart) {
                        this._moveNode(nid, startPos.x + dx, startPos.y + dy);
                    }
                } else {
                    this._moveNode(this.dragNode, x, y);
                }
                return;
            }
            if (this.pendingConn) {
                this._updatePendingConnection(e);
            }
        });

        window.addEventListener('mouseup', e => {
            if (this.isPanning) {
                this.isPanning = false;
                this.container.style.cursor = '';

                if (e.button === 2 && this._rightClickStart) {
                    const dx = e.clientX - this._rightClickStart.x;
                    const dy = e.clientY - this._rightClickStart.y;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist < 5 && this._isBackground(e.target)) {
                        this.panX = this._panBeforeRightClick.x;
                        this.panY = this._panBeforeRightClick.y;
                        this._applyTransform();
                        this._showCreateMenu(e.clientX, e.clientY);
                    }
                    this._rightClickStart = null;
                    this._panBeforeRightClick = null;
                }
            }
            if (this.resizeState) {
                this.resizeState = null;
                this.container.style.cursor = '';
                this._dispatchChange();
                return;
            }
            if (this.marqueeState) {
                this._finishMarquee();
                return;
            }
            if (this.dragNode) {
                this.dragNode = null;
                this._groupDragStart = null;
                this._dispatchChange();
            }
            if (this.pendingConn) {
                this._finishPendingConnection(e);
            }
        });

        window.addEventListener('keydown', e => {
            const hasModifier = e.ctrlKey || e.metaKey;
            if (hasModifier && !this._isEditableTarget(e.target)) {
                const lowerKey = String(e.key || '').toLowerCase();
                if (lowerKey === 'z' && e.shiftKey) {
                    this._redo();
                    e.preventDefault();
                    return;
                }
                if (lowerKey === 'z' && !e.shiftKey) {
                    this._undo();
                    e.preventDefault();
                    return;
                }
                if (lowerKey === 'c') {
                    const copied = this._copySelection();
                    if (copied) e.preventDefault();
                    return;
                }
                if (lowerKey === 'v') {
                    const pasted = this._pasteSelection();
                    if (pasted) e.preventDefault();
                    return;
                }
            }
            if (e.key === 'Escape' && this._contextMenu) {
                this._closeCreateMenu();
                return;
            }
            if (e.key === 'Delete' || e.key === 'Backspace') {
                if (this.selectedConnection) {
                    this._pushUndoState();
                    this.removeConnection(this.selectedConnection);
                    this.selectedConnection = null;
                    this._dispatchChange();
                    return;
                }
                const selectedNodeIds = this._getSelectedNodeIds();
                if (selectedNodeIds.length === 0) return;
                this._pushUndoState();
                for (const nodeId of selectedNodeIds) {
                    this.removeNode(nodeId);
                }
                this._clearSelection();
                this._dispatchChange();
            }
        });

        this.container.addEventListener('dblclick', e => {
            if (this._isBackground(e.target)) {
                this._showCreateMenu(e.clientX, e.clientY);
            }
        });

        // Mouse wheel zoom
        this.container.addEventListener('wheel', e => {
            if (this._contextMenu && this._contextMenu.contains(e.target)) {
                return;
            }
            e.preventDefault();
            if (this._contextMenu) this._closeCreateMenu();
            const rect = this.container.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;
            const factor = e.deltaY > 0 ? 0.9 : 1.1;
            const newScale = Math.max(this.minScale, Math.min(this.maxScale, this.scale * factor));
            this.panX = mouseX - (mouseX - this.panX) * (newScale / this.scale);
            this.panY = mouseY - (mouseY - this.panY) * (newScale / this.scale);
            this.scale = newScale;
            this._applyTransform();
        }, { passive: false });

        // Zoom buttons via event delegation
        this.container.addEventListener('click', e => {
            const btn = e.target.closest('[data-zoom]');
            if (!btn) return;
            const action = btn.dataset.zoom;
            if (action === 'in') this.zoomIn();
            else if (action === 'out') this.zoomOut();
            else if (action === 'reset') this.zoomReset();
        });
    }

    // ---- Transform ----

    _applyTransform() {
        const transform = `translate(${this.panX}px, ${this.panY}px) scale(${this.scale})`;
        this.canvas.style.transform = transform;
        this.canvas.style.transformOrigin = '0 0';
        this.svg.style.transform = transform;
        this.svg.style.transformOrigin = '0 0';
        const gridSize = 20 * this.scale;
        this.container.style.backgroundSize = `${gridSize}px ${gridSize}px`;
        this.container.style.backgroundPosition = `${this.panX}px ${this.panY}px`;
    }

    _getPortCenter(nodeId, portName) {
        const node = this.nodes.get(nodeId);
        if (!node) return null;
        const el = this.canvas.querySelector(`[data-node-id="${nodeId}"]`);
        if (!el) return null;
        const port = el.querySelector(`.ng-port[data-port-name="${portName}"]`);
        if (!port) return null;

        const portRect = port.getBoundingClientRect();
        const elRect = el.getBoundingClientRect();
        const offsetX = ((portRect.left + portRect.width / 2) - elRect.left) / this.scale;
        const offsetY = ((portRect.top + portRect.height / 2) - elRect.top) / this.scale;
        return { x: node.x + offsetX, y: node.y + offsetY };
    }

    // ---- Node Management ----

    addNode(config) {
        const type = config.type || 'step';
        const nodeDef = NodeRegistry.get(type);
        const defaults = nodeDef?.defaultConfig || {};
        const ports = nodeDef?.ports || [
            { name: 'in', dir: 'in', type: 'any' },
            { name: 'out', dir: 'out', type: 'any' }
        ];

        const portValues = {};
        for (const p of ports) {
            if (p.dir === 'in' && p.defaultValue !== undefined) {
                portValues[p.name] = p.defaultValue;
            }
        }
        if (config.portValues) {
            Object.assign(portValues, config.portValues);
        }

        const id = String(config.id || '') || `node-${this.nextNodeId++}`;
        const node = {
            id,
            type,
            title: config.title || defaults.title || 'Step',
            tool: config.tool || defaults.tool || '',
            description: config.description || defaults.description || '',
            x: config.x || 0,
            y: config.y || 0,
            width: config.width || 180,
            height: config.height || null,
            status: config.status || defaults.status || 'pending',
            stepNum: config.stepNum || defaults.stepNum || '',
            resultText: config.resultText || defaults.resultText || '',
            portValues
        };
        this.nodes.set(id, node);
        this._renderNode(node);
        return id;
    }

    removeNode(nodeId) {
        const toRemove = [];
        for (const [connId, conn] of this.connections) {
            if (conn.from === nodeId || conn.to === nodeId) {
                toRemove.push(connId);
            }
        }
        for (const connId of toRemove) {
            this.removeConnection(connId);
        }
        const el = this.canvas.querySelector(`[data-node-id="${nodeId}"]`);
        if (el) el.remove();
        this.nodes.delete(nodeId);
        this.selectedNodes.delete(nodeId);
        if (this.selectedNode === nodeId) this.selectedNode = null;
    }

    _renderNode(node) {
        const nodeDef = NodeRegistry.get(node.type);
        const helpers = { escapeHtml: str => this._escapeHtml(str) };

        const el = nodeDef
            ? nodeDef.render(node, helpers)
            : this._defaultRenderNode(node);

        el.className = `ng-node ng-status-${node.status}`;
        el.dataset.nodeId = node.id;
        el.style.left = node.x + 'px';
        el.style.top = node.y + 'px';
        el.style.width = node.width + 'px';
        if (node.height) {
            el.style.minHeight = node.height + 'px';
        }

        el.addEventListener('click', e => {
            if (e.target.closest('.ng-interactive')) return;
            e.stopPropagation();
            const isToggle = e.ctrlKey || e.metaKey;
            if (isToggle) {
                this._toggleNodeSelection(node.id);
                return;
            }
            this._selectSingleNode(node.id);
        });

        el.querySelectorAll('.ng-interactive').forEach(interactive => {
            interactive.addEventListener('mousedown', e => e.stopPropagation());
            interactive.addEventListener('keydown', e => e.stopPropagation());
        });

        const dragHandle = nodeDef?.getDragHandle
            ? nodeDef.getDragHandle(el)
            : el.querySelector('.ng-node-header');
        if (dragHandle) {
            dragHandle.addEventListener('mousedown', e => {
                if (e.button !== 0) return;
                e.stopPropagation();
                this._pushUndoState();
                const rect = this.container.getBoundingClientRect();
                this.dragNode = node.id;
                this.dragOffsetX = (e.clientX - rect.left - this.panX) / this.scale - node.x;
                this.dragOffsetY = (e.clientY - rect.top - this.panY) / this.scale - node.y;

                if (this.selectedNodes.has(node.id) && this.selectedNodes.size > 1) {
                    this._groupDragStart = new Map();
                    for (const nid of this.selectedNodes) {
                        const n = this.nodes.get(nid);
                        if (n) this._groupDragStart.set(nid, { x: n.x, y: n.y });
                    }
                } else {
                    this._groupDragStart = null;
                }
            });
        }

        el.querySelectorAll('.ng-port').forEach(port => {
            port.addEventListener('mousedown', e => {
                if (e.button === 2) {
                    e.stopPropagation();
                    e.preventDefault();
                    const portName = port.dataset.portName;
                    const portDir = port.dataset.portDir;
                    const nodeId = port.dataset.nodeId;
                    this._startPendingConnection(nodeId, portName, portDir, e, 'ref');
                    return;
                }
                if (e.button !== 0) return;
                e.stopPropagation();
                e.preventDefault();
                const portName = port.dataset.portName;
                const portDir = port.dataset.portDir;
                const nodeId = port.dataset.nodeId;
                this._startPendingConnection(nodeId, portName, portDir, e, 'flow');
            });
        });

        el.querySelectorAll('.ng-port-default').forEach(input => {
            const getPortRef = (target) =>
                target.closest('.ng-port-field')?.dataset.portRef || target.dataset.portRef;
            input.addEventListener('change', e => {
                const portRef = getPortRef(e.target);
                if (portRef) {
                    const val = e.target.type === 'number' ? parseFloat(e.target.value) : e.target.value;
                    this._onDefaultValueChange(node.id, portRef, val);
                }
            });
            input.addEventListener('input', e => {
                const portRef = getPortRef(e.target);
                if (portRef) {
                    const val = e.target.type === 'number' ? parseFloat(e.target.value) : e.target.value;
                    this._onDefaultValueChange(node.id, portRef, val);
                }
            });
        });

        const titleEl = el.querySelector('.ng-node-title');
        if (titleEl) {
            titleEl.addEventListener('dblclick', e => {
                e.stopPropagation();
                this._startTitleEdit(node.id, titleEl);
            });
        }

        this._addResizeHandles(el, node);

        this.canvas.appendChild(el);

        if (nodeDef?.afterRender) {
            nodeDef.afterRender(el, node, {
                ...helpers,
                getNode: () => this.nodes.get(node.id),
                rerender: (n) => {
                    const existing = this.canvas.querySelector(`[data-node-id="${n.id}"]`);
                    if (existing) existing.remove();
                    this._renderNode(n);
                    this._updateConnectionsForNode(n.id);
                },
                updateConnections: (n) => this._updateConnectionsForNode(n.id)
            });
        }
    }

    _defaultRenderNode(node) {
        const el = document.createElement('div');
        el.innerHTML = `
            <div class="ng-ports-in-row" data-node-id="${node.id}">
                <div class="ng-port ng-port-in" data-port-name="in" data-port-dir="in" data-port-type="any" data-node-id="${node.id}"></div>
            </div>
            <div class="ng-node-header">
                <span class="ng-node-num">${node.stepNum}</span>
                <span class="ng-node-title">${this._escapeHtml(node.title)}</span>
            </div>
            <div class="ng-node-body">${this._escapeHtml(node.tool)}</div>
            <div class="ng-node-progress"></div>
            <div class="ng-ports-out-row" data-node-id="${node.id}">
                <div class="ng-port ng-port-out" data-port-name="out" data-port-dir="out" data-port-type="any" data-node-id="${node.id}"></div>
            </div>
        `;
        return el;
    }

    _moveNode(nodeId, x, y) {
        const node = this.nodes.get(nodeId);
        if (!node) return;
        node.x = x;
        node.y = y;
        const el = this.canvas.querySelector(`[data-node-id="${nodeId}"]`);
        if (el) {
            el.style.left = x + 'px';
            el.style.top = y + 'px';
        }
        this._updateConnectionsForNode(nodeId);
    }

    _addResizeHandles(el, node) {
        const dirs = ['nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w'];
        for (const dir of dirs) {
            const handle = document.createElement('div');
            handle.className = `ng-resize-handle ng-resize-${dir}`;
            handle.addEventListener('mousedown', e => {
                if (e.button !== 0) return;
                e.stopPropagation();
                e.preventDefault();
                this._pushUndoState();
                const rect = this.container.getBoundingClientRect();
                const gx = (e.clientX - rect.left - this.panX) / this.scale;
                const gy = (e.clientY - rect.top - this.panY) / this.scale;
                this.resizeState = {
                    nodeId: node.id,
                    handle: dir,
                    startGX: gx,
                    startGY: gy,
                    startW: node.width,
                    startH: el.offsetHeight,
                    startNodeX: node.x,
                    startNodeY: node.y
                };
            });
            el.appendChild(handle);
        }
    }

    _handleResize(e) {
        const rs = this.resizeState;
        if (!rs) return;

        const rect = this.container.getBoundingClientRect();
        const gx = (e.clientX - rect.left - this.panX) / this.scale;
        const gy = (e.clientY - rect.top - this.panY) / this.scale;
        const dx = gx - rs.startGX;
        const dy = gy - rs.startGY;

        const MIN_W = 100;
        const MIN_H = 40;
        const dir = rs.handle;

        let newW = rs.startW;
        let newH = rs.startH;
        let newX = rs.startNodeX;
        let newY = rs.startNodeY;
        let heightExplicit = false;

        const affectsRight = dir === 'e' || dir === 'se' || dir === 'ne';
        const affectsLeft = dir === 'w' || dir === 'sw' || dir === 'nw';
        const affectsBottom = dir === 's' || dir === 'se' || dir === 'sw';
        const affectsTop = dir === 'n' || dir === 'ne' || dir === 'nw';

        if (affectsRight) {
            newW = Math.max(MIN_W, rs.startW + dx);
        }
        if (affectsLeft) {
            newW = Math.max(MIN_W, rs.startW - dx);
            newX = rs.startNodeX + (rs.startW - newW);
        }
        if (affectsBottom) {
            newH = Math.max(MIN_H, rs.startH + dy);
            heightExplicit = true;
        }
        if (affectsTop) {
            newH = Math.max(MIN_H, rs.startH - dy);
            newY = rs.startNodeY + (rs.startH - newH);
            heightExplicit = true;
        }

        const node = this.nodes.get(rs.nodeId);
        if (!node) return;

        node.width = newW;
        node.x = newX;
        node.y = newY;
        if (heightExplicit) {
            node.height = newH;
        }

        const el = this.canvas.querySelector(`[data-node-id="${rs.nodeId}"]`);
        if (el) {
            el.style.width = newW + 'px';
            el.style.left = newX + 'px';
            el.style.top = newY + 'px';
            if (heightExplicit) {
                el.style.minHeight = newH + 'px';
            }

            const ta = el.querySelector('textarea.ng-input-node-field');
            if (ta) {
                ta.style.height = '';
                ta.style.minHeight = '';
            }
        }

        this._updateConnectionsForNode(rs.nodeId);

        const cursorMap = {
            nw: 'nw-resize', n: 'n-resize', ne: 'ne-resize',
            e: 'e-resize', se: 'se-resize', s: 's-resize',
            sw: 'sw-resize', w: 'w-resize'
        };
        this.container.style.cursor = cursorMap[dir] || '';
    }

    resetRunningNodes() {
        for (const [id, node] of this.nodes) {
            if (node.status === 'running' || node.status === 'pending') {
                this.setNodeStatus(id, 'stopped');
            }
        }
    }

    refreshTextareas() {
        if (!this.canvas) return;
        this.canvas.querySelectorAll('textarea.ng-input-node-field').forEach(ta => {
            ta.style.height = 'auto';
            void ta.offsetHeight;
            ta.style.height = ta.scrollHeight + 'px';
        });
        for (const [id] of this.nodes) {
            this._updateConnectionsForNode(id);
        }
    }

    getLayout() {
        const layout = {};
        for (const [id, node] of this.nodes) {
            layout[id] = { x: node.x, y: node.y, width: node.width, height: node.height };
        }
        return layout;
    }

    applyLayout(layout, { skipPosition = false } = {}) {
        if (!layout) return;
        for (const [id, dims] of Object.entries(layout)) {
            const node = this.nodes.get(id);
            if (!node) continue;
            if (dims.width) node.width = dims.width;
            if (dims.height) node.height = dims.height;
            if (!skipPosition) {
                if (dims.x !== undefined) node.x = dims.x;
                if (dims.y !== undefined) node.y = dims.y;
            }
            const el = this.canvas.querySelector(`[data-node-id="${id}"]`);
            if (el) {
                el.style.width = node.width + 'px';
                if (node.height) el.style.minHeight = node.height + 'px';
                if (!skipPosition) {
                    el.style.left = node.x + 'px';
                    el.style.top = node.y + 'px';
                }
            }
        }
        this.refreshConnections();
    }

    setNodeStatus(nodeId, status) {
        const node = this.nodes.get(nodeId);
        if (!node) return;
        node.status = status;
        const el = this.canvas.querySelector(`[data-node-id="${nodeId}"]`);
        if (!el) return;

        const nodeDef = NodeRegistry.get(node.type);
        if (nodeDef?.updateStatus) {
            nodeDef.updateStatus(el, status);
        } else {
            const wasSelected = el.classList.contains('ng-node-selected');
            el.className = `ng-node ng-status-${status}`;
            if (wasSelected) el.classList.add('ng-node-selected');
        }
    }

    setNodeResult(nodeId, resultText) {
        const node = this.nodes.get(nodeId);
        if (!node) return;
        node.resultText = resultText;
        const el = this.canvas.querySelector(`[data-node-id="${nodeId}"]`);
        if (!el) return;
        const nodeDef = NodeRegistry.get(node.type);
        if (nodeDef?.updateResult) {
            nodeDef.updateResult(el, resultText);
        }
    }

    // ---- Default Value Fields ----

    _hideDefaultField(nodeId, portName) {
        const el = this.canvas.querySelector(`[data-node-id="${nodeId}"]`);
        if (!el) return;
        const field = el.querySelector(`.ng-port-field[data-port-ref="${portName}"]`);
        if (field) field.classList.add('ng-port-connected');
    }

    _showDefaultField(nodeId, portName) {
        const el = this.canvas.querySelector(`[data-node-id="${nodeId}"]`);
        if (!el) return;
        const field = el.querySelector(`.ng-port-field[data-port-ref="${portName}"]`);
        if (field) field.classList.remove('ng-port-connected');
    }

    _onDefaultValueChange(nodeId, portName, value) {
        const node = this.nodes.get(nodeId);
        if (!node) return;
        if (!node.portValues) node.portValues = {};
        node.portValues[portName] = value;
        this._dispatchChange();
    }

    _syncDefaultFieldVisibility(nodeId) {
        const ports = this._getPortsForNode(nodeId);
        const inPorts = ports.filter(p => p.dir === 'in');
        for (const p of inPorts) {
            const isFlowConnected = Array.from(this.connections.values()).some(
                c => c.to === nodeId && c.toPort === p.name && c.type !== 'ref'
            );
            if (isFlowConnected) {
                this._hideDefaultField(nodeId, p.name);
            } else {
                this._showDefaultField(nodeId, p.name);
            }
        }
    }

    // ---- Connection Management ----

    addConnection(fromNodeId, fromPort, toNodeId, toPort, type = 'flow') {
        if (fromNodeId === toNodeId) return null;

        // Duplicate check including port names and type
        for (const conn of this.connections.values()) {
            if (conn.from === fromNodeId && conn.fromPort === fromPort &&
                conn.to === toNodeId && conn.toPort === toPort &&
                conn.type === type) return null;
        }

        if (type === 'flow') {
            // Type compatibility check (flow only)
            const fromPorts = this._getPortsForNode(fromNodeId);
            const toPorts = this._getPortsForNode(toNodeId);
            const fromPortDef = fromPorts.find(p => p.name === fromPort);
            const toPortDef = toPorts.find(p => p.name === toPort);
            if (fromPortDef && toPortDef) {
                if (!this._isTypeCompatible(fromPortDef.type, toPortDef.type)) return null;
            }

            // Replace existing flow connection on this input port
            for (const [connId, conn] of this.connections) {
                if (conn.to === toNodeId && conn.toPort === toPort && conn.type === 'flow') {
                    this.removeConnection(connId);
                    break;
                }
            }
        }

        if (type === 'ref') {
            if (!this._allowsRef(toNodeId)) return null;
            const maxAttachments = parseInt(localStorage.getItem('maxAttachments') || '5');
            const currentRefCount = Array.from(this.connections.values())
                .filter(c => c.to === toNodeId && c.toPort === toPort && c.type === 'ref').length;
            if (currentRefCount >= maxAttachments) return null;
        }

        const id = `conn-${this.nextConnId++}`;
        const conn = { id, from: fromNodeId, fromPort, to: toNodeId, toPort, type };
        this.connections.set(id, conn);
        this._renderConnection(conn);
        if (type === 'flow') {
            this._hideDefaultField(toNodeId, toPort);
            this._resolveOutputTypes(toNodeId);
        }
        return id;
    }

    removeConnection(connId) {
        const conn = this.connections.get(connId);
        this._removeConnectionElement(connId);
        this.connections.delete(connId);
        if (conn && conn.type !== 'ref') {
            const stillFlowConnected = Array.from(this.connections.values()).some(
                c => c.to === conn.to && c.toPort === conn.toPort && c.type !== 'ref'
            );
            if (!stillFlowConnected) {
                this._showDefaultField(conn.to, conn.toPort);
            }
            this._resolveOutputTypes(conn.to);
        }
    }

    _renderConnection(conn) {
        const hitPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        hitPath.dataset.connId = conn.id;
        hitPath.classList.add('ng-connection-hit');

        const visPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        visPath.dataset.connId = conn.id;
        visPath.classList.add('ng-connection');

        if (conn.type === 'ref') {
            visPath.classList.add('ng-conn-ref');
        }

        const fromPorts = this._getPortsForNode(conn.from);
        const fromPortDef = fromPorts.find(p => p.name === conn.fromPort);
        if (fromPortDef && fromPortDef.type !== 'any') {
            visPath.classList.add(`ng-conn-type-${fromPortDef.type}`);
        }

        const selectHandler = e => {
            e.stopPropagation();
            this._clearSelection();
            this.selectedConnection = conn.id;
            visPath.classList.add('ng-conn-selected');
        };
        hitPath.addEventListener('click', selectHandler);
        visPath.addEventListener('click', selectHandler);

        hitPath.addEventListener('contextmenu', e => {
            e.preventDefault();
            e.stopPropagation();
            this._pushUndoState();
            this.removeConnection(conn.id);
            this._dispatchChange();
        });

        this.svg.appendChild(hitPath);
        this.svg.appendChild(visPath);
        this._updateConnectionPath(conn);
    }

    _removeConnectionElement(connId) {
        this.svg.querySelectorAll(`[data-conn-id="${connId}"]`).forEach(el => el.remove());
    }

    _updateConnectionPath(conn) {
        const from = this._getPortCenter(conn.from, conn.fromPort);
        const to = this._getPortCenter(conn.to, conn.toPort);
        if (!from || !to) return;

        const dy = Math.abs(to.y - from.y) * 0.5;
        const d = `M ${from.x} ${from.y} C ${from.x} ${from.y + dy}, ${to.x} ${to.y - dy}, ${to.x} ${to.y}`;

        const paths = this.svg.querySelectorAll(`[data-conn-id="${conn.id}"]`);
        paths.forEach(p => p.setAttribute('d', d));
    }

    _updateConnectionsForNode(nodeId) {
        for (const conn of this.connections.values()) {
            if (conn.from === nodeId || conn.to === nodeId) {
                this._updateConnectionPath(conn);
            }
        }
    }

    refreshConnections() {
        for (const conn of this.connections.values()) {
            this._updateConnectionPath(conn);
        }
    }

    // ---- Pending Connection (drag from port) ----

    _startPendingConnection(fromNodeId, portName, portDir, e, connType = 'flow') {
        this._undoPushedForPendingConn = false;
        let existingConn = null;
        if (connType === 'flow' && portDir === 'in') {
            for (const [connId, conn] of this.connections) {
                if (conn.to === fromNodeId && conn.toPort === portName && conn.type !== 'ref') {
                    existingConn = {
                        connId,
                        otherNode: conn.from,
                        otherPort: conn.fromPort,
                        reconnectDir: 'out'
                    };
                    break;
                }
            }
        }

        let anchorNodeId, anchorPort, anchorDir;
        if (existingConn) {
            this._pushUndoState();
            this._undoPushedForPendingConn = true;
            this.removeConnection(existingConn.connId);
            anchorNodeId = existingConn.otherNode;
            anchorPort = existingConn.otherPort;
            anchorDir = existingConn.reconnectDir;
        } else {
            anchorNodeId = fromNodeId;
            anchorPort = portName;
            anchorDir = portDir;
        }

        const center = this._getPortCenter(anchorNodeId, anchorPort);
        if (!center) return;

        const tempLine = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        tempLine.classList.add('ng-connection', connType === 'ref' ? 'ng-conn-pending-ref' : 'ng-conn-pending');
        this.svg.appendChild(tempLine);

        const ports = this._getPortsForNode(anchorNodeId);
        const portDef = ports.find(p => p.name === anchorPort);
        const portType = portDef?.type || 'any';

        this.pendingConn = {
            fromNodeId: anchorNodeId,
            fromPort: anchorPort,
            fromDir: anchorDir,
            fromType: portType,
            connType,
            startX: center.x,
            startY: center.y,
            tempLine
        };
        this._highlightCompatiblePorts();
    }

    _findSnapPort(mx, my) {
        if (!this.pendingConn) return null;
        const SNAP_RADIUS = 15;
        const targetDir = this.pendingConn.fromDir === 'out' ? 'in' : 'out';
        let best = null, bestDist = SNAP_RADIUS;

        for (const [nodeId, node] of this.nodes) {
            if (nodeId === this.pendingConn.fromNodeId) continue;
            const el = this.canvas.querySelector(`[data-node-id="${nodeId}"]`);
            if (!el) continue;

            const ports = this._getPortsForNode(nodeId);
            for (const p of ports) {
                if (p.dir !== targetDir) continue;

                let compatible;
                if (this.pendingConn.connType === 'ref') {
                    if (!this._allowsRef(nodeId)) continue;
                    const maxAtt = parseInt(localStorage.getItem('maxAttachments') || '5');
                    const refCount = Array.from(this.connections.values())
                        .filter(c => c.to === nodeId && c.toPort === p.name && c.type === 'ref').length;
                    compatible = refCount < maxAtt;
                } else if (this.pendingConn.fromDir === 'out') {
                    compatible = this._isTypeCompatible(this.pendingConn.fromType, p.type);
                } else {
                    compatible = this._isTypeCompatible(p.type, this.pendingConn.fromType);
                }
                if (!compatible) continue;

                const center = this._getPortCenter(nodeId, p.name);
                if (!center) continue;
                const dist = Math.hypot(mx - center.x, my - center.y);
                if (dist < bestDist) {
                    bestDist = dist;
                    best = { nodeId, portName: p.name, center };
                }
            }
        }
        return best;
    }

    _updatePendingConnection(e) {
        if (!this.pendingConn) return;
        const rect = this.container.getBoundingClientRect();
        const mx = (e.clientX - rect.left - this.panX) / this.scale;
        const my = (e.clientY - rect.top - this.panY) / this.scale;

        const snap = this._findSnapPort(mx, my);

        const prevSnap = this.pendingConn._snapTarget;
        if (prevSnap) {
            const prevEl = this.canvas.querySelector(
                `[data-node-id="${prevSnap.nodeId}"] .ng-port[data-port-name="${prevSnap.portName}"]`);
            if (prevEl) prevEl.classList.remove('ng-port-snap');
        }

        let ex = mx, ey = my;
        if (snap) {
            ex = snap.center.x;
            ey = snap.center.y;
            const snapEl = this.canvas.querySelector(
                `[data-node-id="${snap.nodeId}"] .ng-port[data-port-name="${snap.portName}"]`);
            if (snapEl) snapEl.classList.add('ng-port-snap');
            this.pendingConn._snapTarget = snap;
        } else {
            this.pendingConn._snapTarget = null;
        }

        const { startX: x1, startY: y1 } = this.pendingConn;
        const dy = Math.abs(ey - y1) * 0.5;
        const d = `M ${x1} ${y1} C ${x1} ${y1 + dy}, ${ex} ${ey - dy}, ${ex} ${ey}`;
        this.pendingConn.tempLine.setAttribute('d', d);
    }

    _finishPendingConnection(e) {
        if (!this.pendingConn) return;
        this.pendingConn.tempLine.remove();

        const rect = this.container.getBoundingClientRect();
        const mx = (e.clientX - rect.left - this.panX) / this.scale;
        const my = (e.clientY - rect.top - this.panY) / this.scale;

        const targetDir = this.pendingConn.fromDir === 'out' ? 'in' : 'out';
        let bestPort = null;
        let bestDist = Infinity;

        for (const [nodeId, node] of this.nodes) {
            if (nodeId === this.pendingConn.fromNodeId) continue;
            const el = this.canvas.querySelector(`[data-node-id="${nodeId}"]`);
            if (!el) continue;

            const PAD = 15;
            if (mx < node.x - PAD || mx > node.x + el.offsetWidth + PAD ||
                my < node.y - PAD || my > node.y + el.offsetHeight + PAD) continue;

            const ports = this._getPortsForNode(nodeId);
            for (const p of ports) {
                if (p.dir !== targetDir) continue;

                let compatible;
                if (this.pendingConn.connType === 'ref') {
                    if (!this._allowsRef(nodeId)) continue;
                    const maxAtt = parseInt(localStorage.getItem('maxAttachments') || '5');
                    const refCount = Array.from(this.connections.values())
                        .filter(c => c.to === nodeId && c.toPort === p.name && c.type === 'ref').length;
                    compatible = refCount < maxAtt;
                } else if (this.pendingConn.fromDir === 'out') {
                    compatible = this._isTypeCompatible(this.pendingConn.fromType, p.type);
                } else {
                    compatible = this._isTypeCompatible(p.type, this.pendingConn.fromType);
                }
                if (!compatible) continue;

                const center = this._getPortCenter(nodeId, p.name);
                if (!center) continue;
                const dist = Math.hypot(mx - center.x, my - center.y);
                if (dist < bestDist) {
                    bestDist = dist;
                    bestPort = { nodeId, portName: p.name };
                }
            }
        }

        if (bestPort) {
            if (!this._undoPushedForPendingConn) {
                this._pushUndoState();
            }
            const ct = this.pendingConn.connType || 'flow';
            if (this.pendingConn.fromDir === 'out') {
                this.addConnection(
                    this.pendingConn.fromNodeId, this.pendingConn.fromPort,
                    bestPort.nodeId, bestPort.portName, ct
                );
            } else {
                this.addConnection(
                    bestPort.nodeId, bestPort.portName,
                    this.pendingConn.fromNodeId, this.pendingConn.fromPort, ct
                );
            }
            this._dispatchChange();
        }
        this._clearPortHighlights();
        this.pendingConn = null;
    }

    _highlightCompatiblePorts() {
        if (!this.pendingConn) return;
        const { fromNodeId, fromDir, fromType, connType } = this.pendingConn;
        const targetDir = fromDir === 'out' ? 'in' : 'out';

        this.canvas.querySelectorAll('.ng-port').forEach(portEl => {
            const portNodeId = portEl.dataset.nodeId;
            const portDir = portEl.dataset.portDir;
            const portType = portEl.dataset.portType;

            if (portNodeId === fromNodeId) {
                portEl.classList.add('ng-port-dimmed');
                return;
            }

            if (portDir !== targetDir) {
                portEl.classList.add('ng-port-dimmed');
                return;
            }

            let compatible;
            if (connType === 'ref') {
                if (!this._allowsRef(portNodeId)) {
                    portEl.classList.add('ng-port-dimmed');
                    return;
                }
                const maxAtt = parseInt(localStorage.getItem('maxAttachments') || '5');
                const portName = portEl.dataset.portName;
                const refCount = Array.from(this.connections.values())
                    .filter(c => c.to === portNodeId && c.toPort === portName && c.type === 'ref').length;
                compatible = refCount < maxAtt;
            } else if (fromDir === 'out') {
                compatible = this._isTypeCompatible(fromType, portType);
            } else {
                compatible = this._isTypeCompatible(portType, fromType);
            }

            if (compatible) {
                portEl.classList.add('ng-port-compatible');
            } else {
                portEl.classList.add('ng-port-dimmed');
            }
        });
    }

    _clearPortHighlights() {
        this.canvas.querySelectorAll('.ng-port-compatible, .ng-port-dimmed, .ng-port-snap').forEach(el => {
            el.classList.remove('ng-port-compatible', 'ng-port-dimmed', 'ng-port-snap');
        });
    }

    // ---- Selection ----

    _clearSelection() {
        if (this.selectedConnection) {
            this.svg.querySelectorAll(`[data-conn-id="${this.selectedConnection}"]`).forEach(p => {
                p.classList.remove('ng-conn-selected');
            });
            this.selectedConnection = null;
        }
        for (const nodeId of this.selectedNodes) {
            const el = this.canvas.querySelector(`[data-node-id="${nodeId}"]`);
            if (el) el.classList.remove('ng-node-selected');
        }
        this.selectedNodes.clear();
        this.selectedNode = null;
    }

    _setNodeSelected(nodeId, selected) {
        const el = this.canvas.querySelector(`[data-node-id="${nodeId}"]`);
        if (!el) return;
        el.classList.toggle('ng-node-selected', selected);
        if (selected) this.selectedNodes.add(nodeId);
        else this.selectedNodes.delete(nodeId);
        this.selectedNode = this.selectedNodes.size === 1 ? Array.from(this.selectedNodes)[0] : null;
    }

    _selectSingleNode(nodeId) {
        this._clearSelection();
        this._setNodeSelected(nodeId, true);
    }

    _toggleNodeSelection(nodeId) {
        const alreadySelected = this.selectedNodes.has(nodeId);
        this._setNodeSelected(nodeId, !alreadySelected);
    }

    _getSelectedNodeIds() {
        if (this.selectedNodes.size > 0) return Array.from(this.selectedNodes);
        if (this.selectedNode) return [this.selectedNode];
        return [];
    }

    _isEditableTarget(target) {
        if (!target || !(target instanceof Element)) return false;
        if (target.closest('.ng-create-menu')) return true;
        const editable = target.closest('input, textarea, select, [contenteditable=""], [contenteditable="true"]');
        return Boolean(editable);
    }

    _startMarquee(e, append = false) {
        const rect = this.container.getBoundingClientRect();
        const startX = e.clientX - rect.left;
        const startY = e.clientY - rect.top;

        if (!append) this._clearSelection();

        const box = document.createElement('div');
        box.className = 'ng-marquee';
        box.style.left = startX + 'px';
        box.style.top = startY + 'px';
        box.style.width = '0px';
        box.style.height = '0px';
        this.container.appendChild(box);

        this.marqueeState = {
            startX,
            startY,
            currentX: startX,
            currentY: startY,
            append,
            box
        };
    }

    _updateMarquee(e) {
        if (!this.marqueeState) return;
        const rect = this.container.getBoundingClientRect();
        this.marqueeState.currentX = e.clientX - rect.left;
        this.marqueeState.currentY = e.clientY - rect.top;

        const minX = Math.min(this.marqueeState.startX, this.marqueeState.currentX);
        const minY = Math.min(this.marqueeState.startY, this.marqueeState.currentY);
        const width = Math.abs(this.marqueeState.currentX - this.marqueeState.startX);
        const height = Math.abs(this.marqueeState.currentY - this.marqueeState.startY);

        this.marqueeState.box.style.left = minX + 'px';
        this.marqueeState.box.style.top = minY + 'px';
        this.marqueeState.box.style.width = width + 'px';
        this.marqueeState.box.style.height = height + 'px';
    }

    _finishMarquee() {
        if (!this.marqueeState) return;

        const { startX, startY, currentX, currentY, append, box } = this.marqueeState;
        if (box) box.remove();

        const minPX = Math.min(startX, currentX);
        const minPY = Math.min(startY, currentY);
        const maxPX = Math.max(startX, currentX);
        const maxPY = Math.max(startY, currentY);

        const minGX = (minPX - this.panX) / this.scale;
        const minGY = (minPY - this.panY) / this.scale;
        const maxGX = (maxPX - this.panX) / this.scale;
        const maxGY = (maxPY - this.panY) / this.scale;

        if (!append) {
            for (const nodeId of this.selectedNodes) {
                const el = this.canvas.querySelector(`[data-node-id="${nodeId}"]`);
                if (el) el.classList.remove('ng-node-selected');
            }
            this.selectedNodes.clear();
            this.selectedNode = null;
        }

        for (const [nodeId, node] of this.nodes) {
            const el = this.canvas.querySelector(`[data-node-id="${nodeId}"]`);
            if (!el) continue;
            const w = el.offsetWidth || node.width || 180;
            const h = el.offsetHeight || node.height || 60;

            const intersects = node.x < maxGX && node.x + w > minGX && node.y < maxGY && node.y + h > minGY;
            if (intersects) {
                this._setNodeSelected(nodeId, true);
            }
        }

        this.marqueeState = null;
    }

    _getViewportCenterGraphPos() {
        const cx = this.container.clientWidth / 2;
        const cy = this.container.clientHeight / 2;
        return {
            x: (cx - this.panX) / this.scale,
            y: (cy - this.panY) / this.scale
        };
    }

    _copySelection() {
        const selectedNodeIds = this._getSelectedNodeIds();
        if (selectedNodeIds.length === 0) return false;
        const idSet = new Set(selectedNodeIds);

        const nodes = [];
        let minX = Infinity;
        let minY = Infinity;
        let maxX = -Infinity;
        let maxY = -Infinity;

        for (const nodeId of selectedNodeIds) {
            const node = this.nodes.get(nodeId);
            if (!node) continue;
            const el = this.canvas.querySelector(`[data-node-id="${nodeId}"]`);
            const w = el ? el.offsetWidth : (node.width || 180);
            const h = el ? el.offsetHeight : (node.height || 60);

            nodes.push({
                id: node.id,
                type: node.type,
                title: node.title,
                tool: node.tool,
                description: node.description,
                x: node.x,
                y: node.y,
                width: node.width,
                height: node.height,
                status: node.status,
                stepNum: node.stepNum,
                resultText: node.resultText,
                portValues: { ...(node.portValues || {}) }
            });

            minX = Math.min(minX, node.x);
            minY = Math.min(minY, node.y);
            maxX = Math.max(maxX, node.x + w);
            maxY = Math.max(maxY, node.y + h);
        }

        if (nodes.length === 0) return false;

        const connections = [];
        for (const conn of this.connections.values()) {
            if (!idSet.has(conn.from) || !idSet.has(conn.to)) continue;
            connections.push({
                from: conn.from,
                fromPort: conn.fromPort,
                to: conn.to,
                toPort: conn.toPort,
                type: conn.type || 'flow'
            });
        }

        this.clipboard = {
            nodes,
            connections,
            centerX: (minX + maxX) / 2,
            centerY: (minY + maxY) / 2
        };
        return true;
    }

    _pasteSelection() {
        if (!this.clipboard || !Array.isArray(this.clipboard.nodes) || this.clipboard.nodes.length === 0) {
            return false;
        }

        this._pushUndoState();
        const center = this._getViewportCenterGraphPos();
        const deltaX = center.x - this.clipboard.centerX;
        const deltaY = center.y - this.clipboard.centerY;
        const oldToNew = new Map();
        const newIds = [];

        this._clearSelection();

        for (const srcNode of this.clipboard.nodes) {
            const nodeConfig = {
                type: srcNode.type,
                title: srcNode.title,
                tool: srcNode.tool,
                description: srcNode.description,
                x: srcNode.x + deltaX,
                y: srcNode.y + deltaY,
                width: srcNode.width,
                height: srcNode.height,
                status: srcNode.status || 'pending',
                stepNum: srcNode.stepNum,
                resultText: srcNode.resultText || '',
                portValues: { ...(srcNode.portValues || {}) }
            };
            const newId = this.addNode(nodeConfig);
            oldToNew.set(srcNode.id, newId);
            newIds.push(newId);
        }

        for (const srcConn of this.clipboard.connections || []) {
            const fromId = oldToNew.get(srcConn.from);
            const toId = oldToNew.get(srcConn.to);
            if (!fromId || !toId) continue;
            this.addConnection(
                fromId,
                srcConn.fromPort || 'out',
                toId,
                srcConn.toPort || 'in',
                srcConn.type || 'flow'
            );
        }

        for (const nodeId of newIds) {
            this._setNodeSelected(nodeId, true);
        }
        this._dispatchChange();
        return true;
    }

    // ---- Create from Plan ----

    createFromPlan(planData) {
        this.clear();
        const steps = planData.steps || [];
        if (steps.length === 0) return;

        const nodeWidth = 180;
        const edgeGap = 50;
        const estimatedNodeHeight = 60;
        const startX = 0;
        let currentY = 40;

        const promptText = planData.userMessage || planData.goal || '';
        const promptNodeId = this.addNode({
            id: 'prompt-input',
            type: 'string',
            title: 'Prompt',
            x: startX,
            y: currentY,
            status: 'completed',
            portValues: { out: promptText }
        });
        currentY += estimatedNodeHeight + edgeGap;

        const nodeIds = [];
        for (let i = 0; i < steps.length; i++) {
            const step = steps[i];
            const id = this.addNode({
                id: `step-${i + 1}`,
                title: step.name || `Step ${i + 1}`,
                tool: step.tool || '',
                description: step.description || '',
                x: startX,
                y: currentY,
                stepNum: step.id || `${i + 1}`,
                status: 'pending'
            });
            nodeIds.push(id);
            currentY += estimatedNodeHeight + edgeGap;
        }

        if (nodeIds.length > 0) {
            this.addConnection(promptNodeId, 'out', nodeIds[0], 'in');
        }
        for (let i = 0; i < nodeIds.length - 1; i++) {
            this.addConnection(nodeIds[i], 'out', nodeIds[i + 1], 'in');
        }

        this.snapshotExecutionPlanHash();
        this._needsLayout = 'full';
    }

    clear() {
        this.nodes.clear();
        this.connections.clear();
        if (this.canvas) this.canvas.innerHTML = '';
        if (this.svg) this.svg.innerHTML = '';
        this.nextNodeId = 1;
        this.nextConnId = 1;
        this.panX = 0;
        this.panY = 0;
        this.scale = 1;
        this.selectedNode = null;
        this.selectedConnection = null;
        this.selectedNodes.clear();
        this.marqueeState = null;
        this._groupDragStart = null;
        this._undoStack = [];
        this._redoStack = [];
        this._needsLayout = null;
        this._applyTransform();
    }

    // ---- Serialization ----

    getState() {
        return {
            nodes: Array.from(this.nodes.values()),
            connections: Array.from(this.connections.values()),
            panX: this.panX,
            panY: this.panY,
            scale: this.scale
        };
    }

    setState(state) {
        if (!state) return;
        this.clear();
        this.panX = state.panX || 0;
        this.panY = state.panY || 0;
        this.scale = state.scale || 1;
        this._applyTransform();

        for (const n of (state.nodes || [])) {
            this.addNode(n);
        }
        for (const c of (state.connections || [])) {
            const fromPort = c.fromPort || 'out';
            const toPort = c.toPort || 'in';
            this.addConnection(c.from, fromPort, c.to, toPort, c.type || 'flow');
        }
        this._resolveAllOutputTypes();
    }

    // ---- Graph -> Execution Plan ----

    _buildGraphMaps() {
        const connectedIds = new Set();
        const children = new Map();
        const parents = new Map();
        for (const node of this.nodes.values()) {
            children.set(node.id, []);
            parents.set(node.id, []);
        }
        for (const conn of this.connections.values()) {
            if (conn.type === 'ref') continue;
            children.get(conn.from)?.push(conn.to);
            parents.get(conn.to)?.push(conn.from);
            connectedIds.add(conn.from);
            connectedIds.add(conn.to);
        }
        return { children, parents, connectedIds };
    }

    toExecutionPlan(options = {}) {
        if (this.nodes.size === 0) return null;

        const { excludeResult = false } = options;
        const { children, parents, connectedIds } = this._buildGraphMaps();

        const candidateIds = new Set();
        for (const node of this.nodes.values()) {
            if (!connectedIds.has(node.id)) continue;
            const def = NodeRegistry.get(node.type);
            if (def?.dataOnly) continue;
            if (excludeResult && def?.result) continue;
            candidateIds.add(node.id);
        }

        // Topological sort (Kahn's algorithm) over candidates only
        const inDegree = new Map();
        const filteredChildren = new Map();
        const filteredParents = new Map();
        for (const id of candidateIds) {
            filteredChildren.set(id, (children.get(id) || []).filter(c => candidateIds.has(c)));
            filteredParents.set(id, (parents.get(id) || []).filter(p => candidateIds.has(p)));
            inDegree.set(id, filteredParents.get(id).length);
        }

        const queue = [];
        for (const [id, deg] of inDegree) {
            if (deg === 0) queue.push(id);
        }

        const sorted = [];
        while (queue.length > 0) {
            const id = queue.shift();
            sorted.push(id);
            for (const childId of (filteredChildren.get(id) || [])) {
                inDegree.set(childId, inDegree.get(childId) - 1);
                if (inDegree.get(childId) === 0) queue.push(childId);
            }
        }

        // Assign numbering with sub-numbering for parallel branches
        const numbering = new Map();
        let mainCounter = 0;

        for (const nodeId of sorted) {
            const parentIds = filteredParents.get(nodeId) || [];

            if (parentIds.length === 0) {
                mainCounter++;
                numbering.set(nodeId, `${mainCounter}`);
            } else if (parentIds.length === 1) {
                const parentId = parentIds[0];
                const siblings = filteredChildren.get(parentId) || [];
                if (siblings.length > 1) {
                    const siblingIndex = siblings.indexOf(nodeId) + 1;
                    const parentNum = numbering.get(parentId) || `${mainCounter}`;
                    numbering.set(nodeId, `${parentNum}-${siblingIndex}`);
                } else {
                    mainCounter++;
                    numbering.set(nodeId, `${mainCounter}`);
                }
            } else {
                mainCounter++;
                numbering.set(nodeId, `${mainCounter}`);
            }
        }

        const steps = sorted.map(nodeId => {
            const node = this.nodes.get(nodeId);
            const refs = [];
            const inputs = {};
            for (const conn of this.connections.values()) {
                if (conn.type === 'ref' && conn.to === nodeId) {
                    const refNode = this.nodes.get(conn.from);
                    if (refNode) {
                        refs.push({
                            nodeId: refNode.id,
                            title: refNode.title,
                            nodeType: refNode.type,
                            portValues: refNode.portValues
                        });
                    }
                }
                if (conn.type === 'flow' && conn.to === nodeId) {
                    const parentNode = this.nodes.get(conn.from);
                    if (parentNode) {
                        const parentDef = NodeRegistry.get(parentNode.type);
                        if (parentDef?.dataOnly) {
                            inputs[conn.toPort || 'in'] = {
                                nodeType: parentNode.type,
                                title: parentNode.title,
                                portValues: parentNode.portValues
                            };
                        }
                    }
                }
            }
            const step = {
                id: numbering.get(nodeId),
                name: node.title,
                tool: node.tool,
                description: node.description,
                depends_on: (filteredParents.get(nodeId) || []).map(pid => numbering.get(pid))
            };
            if (refs.length > 0) step.references = refs;
            if (Object.keys(inputs).length > 0) step.inputs = inputs;
            return step;
        });

        return {
            goal: '',
            steps
        };
    }

    getExecutionPlanHash() {
        const plan = this.toExecutionPlan({ excludeResult: true });
        if (!plan || !plan.steps || plan.steps.length === 0) return null;
        return plan.steps.map(s => `${s.id}:${s.tool}:${s.name}`).join('|');
    }

    hasExecutionLogicChanged() {
        const newHash = this.getExecutionPlanHash();
        if (this._lastExecutionPlanHash === null) {
            this._lastExecutionPlanHash = newHash;
            return false;
        }
        const changed = newHash !== this._lastExecutionPlanHash;
        if (changed) this._lastExecutionPlanHash = newHash;
        return changed;
    }

    snapshotExecutionPlanHash() {
        this._lastExecutionPlanHash = this.getExecutionPlanHash();
    }

    // ---- Zoom Controls ----

    zoomIn() {
        const cx = this.container.clientWidth / 2;
        const cy = this.container.clientHeight / 2;
        const newScale = Math.min(this.maxScale, this.scale * 1.2);
        this.panX = cx - (cx - this.panX) * (newScale / this.scale);
        this.panY = cy - (cy - this.panY) * (newScale / this.scale);
        this.scale = newScale;
        this._applyTransform();
    }

    zoomOut() {
        const cx = this.container.clientWidth / 2;
        const cy = this.container.clientHeight / 2;
        const newScale = Math.max(this.minScale, this.scale * 0.8);
        this.panX = cx - (cx - this.panX) * (newScale / this.scale);
        this.panY = cy - (cy - this.panY) * (newScale / this.scale);
        this.scale = newScale;
        this._applyTransform();
    }

    zoomReset() {
        this.scale = 1;
        this.panX = 0;
        this.panY = 0;
        this._applyTransform();
    }

    _relayoutVertical(gap = 50) {
        this.refreshTextareas();
        const ordered = ['prompt-input'];
        let i = 1;
        while (this.nodes.has(`step-${i}`)) {
            ordered.push(`step-${i}`);
            i++;
        }
        let currentY = 40;
        for (const id of ordered) {
            const node = this.nodes.get(id);
            if (!node) continue;
            node.y = currentY;
            const el = this.canvas.querySelector(`[data-node-id="${id}"]`);
            if (el) {
                el.style.top = currentY + 'px';
                currentY += el.offsetHeight + gap;
            } else {
                currentY += 120;
            }
        }
        this.refreshConnections();
    }

    fitToView(padding = 40) {
        if (this.nodes.size === 0) return;

        const cw = this.container.clientWidth;
        const ch = this.container.clientHeight;
        if (cw === 0 || ch === 0) return;

        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        for (const node of this.nodes.values()) {
            const el = this.canvas.querySelector(`[data-node-id="${node.id}"]`);
            const w = el ? el.offsetWidth : 180;
            const h = el ? el.offsetHeight : 60;
            if (node.x < minX) minX = node.x;
            if (node.y < minY) minY = node.y;
            if (node.x + w > maxX) maxX = node.x + w;
            if (node.y + h > maxY) maxY = node.y + h;
        }

        const contentW = maxX - minX;
        const contentH = maxY - minY;
        if (contentW <= 0 || contentH <= 0) return;

        const scaleX = (cw - padding * 2) / contentW;
        const scaleY = (ch - padding * 2) / contentH;
        const newScale = Math.min(scaleX, scaleY, 1.0);

        this.scale = Math.max(this.minScale, Math.min(this.maxScale, newScale));
        this.panX = (cw - contentW * this.scale) / 2 - minX * this.scale;
        this.panY = (ch - contentH * this.scale) / 2 - minY * this.scale;
        this._applyTransform();
        this.refreshConnections();
    }

    // ---- Background Check ----

    _isBackground(target) {
        if (target === this.container || target === this.svg || target === this.canvas) return true;
        if (!this.container.contains(target)) return false;
        if (target.closest('.ng-node, .ng-zoom-controls, .ng-create-menu, .graph-popout-btn')) return false;
        if (target instanceof SVGElement && target !== this.svg) return false;
        return true;
    }

    // ---- Create Node Menu ----

    _showCreateMenu(screenX, screenY) {
        this._closeCreateMenu();

        const containerRect = this.container.getBoundingClientRect();
        const graphX = (screenX - containerRect.left - this.panX) / this.scale;
        const graphY = (screenY - containerRect.top - this.panY) / this.scale;
        this._contextMenuGraphPos = { x: graphX, y: graphY };

        const menu = document.createElement('div');
        menu.className = 'ng-create-menu';

        const searchWrap = document.createElement('div');
        searchWrap.className = 'ng-create-menu-search-wrap';
        const searchIcon = document.createElement('span');
        searchIcon.className = 'ng-create-menu-search-icon';
        searchIcon.textContent = '\u{1F50D}';
        const searchInput = document.createElement('input');
        searchInput.type = 'text';
        searchInput.className = 'ng-create-menu-search';
        searchInput.placeholder = 'Search...';
        searchInput.addEventListener('input', () => this._filterCreateMenu(searchInput.value));
        searchInput.addEventListener('keydown', e => e.stopPropagation());
        searchWrap.appendChild(searchIcon);
        searchWrap.appendChild(searchInput);

        const title = document.createElement('div');
        title.className = 'ng-create-menu-title';
        title.textContent = 'Create Node';

        const list = document.createElement('div');
        list.className = 'ng-create-menu-list';

        const categories = NodeRegistry.getByCategory();
        const catNames = Object.keys(categories).sort();

        for (const catName of catNames) {
            const catEl = document.createElement('div');
            catEl.className = 'ng-create-menu-category';
            catEl.dataset.category = catName;

            const catHeader = document.createElement('div');
            catHeader.className = 'ng-create-menu-category-header';
            const catArrow = document.createElement('span');
            catArrow.className = 'ng-create-menu-cat-arrow';
            catArrow.textContent = '\u25B6';
            const catLabel = document.createElement('span');
            catLabel.textContent = catName;
            catHeader.appendChild(catArrow);
            catHeader.appendChild(catLabel);

            const catItems = document.createElement('div');
            catItems.className = 'ng-create-menu-items';

            for (const entry of categories[catName]) {
                const item = document.createElement('div');
                item.className = 'ng-create-menu-item';
                item.textContent = entry.label;
                item.dataset.nodeType = entry.type;
                item.addEventListener('click', () => this._onNodeTypeSelected(entry.type));
                catItems.appendChild(item);
            }

            catHeader.addEventListener('click', () => {
                const expanded = catEl.classList.toggle('ng-cat-expanded');
                catArrow.textContent = expanded ? '\u25BC' : '\u25B6';
            });

            catEl.appendChild(catHeader);
            catEl.appendChild(catItems);
            list.appendChild(catEl);
        }

        menu.appendChild(searchWrap);
        menu.appendChild(title);
        menu.appendChild(list);

        let posX = screenX - containerRect.left;
        let posY = screenY - containerRect.top;

        const menuW = 220;
        const menuH = 300;
        if (posX + menuW > containerRect.width) posX = containerRect.width - menuW - 4;
        if (posY + menuH > containerRect.height) posY = containerRect.height - menuH - 4;
        if (posX < 4) posX = 4;
        if (posY < 4) posY = 4;

        menu.style.left = posX + 'px';
        menu.style.top = posY + 'px';

        this.container.appendChild(menu);
        this._contextMenu = menu;

        requestAnimationFrame(() => searchInput.focus());
    }

    _closeCreateMenu() {
        if (this._contextMenu) {
            this._contextMenu.remove();
            this._contextMenu = null;
        }
    }

    _filterCreateMenu(query) {
        if (!this._contextMenu) return;
        const q = query.toLowerCase().trim();
        const categories = this._contextMenu.querySelectorAll('.ng-create-menu-category');

        for (const catEl of categories) {
            const items = catEl.querySelectorAll('.ng-create-menu-item');
            let anyVisible = false;

            for (const item of items) {
                const match = !q || item.textContent.toLowerCase().includes(q);
                item.style.display = match ? '' : 'none';
                if (match) anyVisible = true;
            }

            catEl.style.display = anyVisible ? '' : 'none';

            if (q && anyVisible) {
                catEl.classList.add('ng-cat-expanded');
                const arrow = catEl.querySelector('.ng-create-menu-cat-arrow');
                if (arrow) arrow.textContent = '\u25BC';
            }
        }
    }

    _onNodeTypeSelected(type) {
        this._pushUndoState();
        const { x, y } = this._contextMenuGraphPos;
        this.addNode({ type, x, y });
        this._closeCreateMenu();
        this._dispatchChange();
    }

    // ---- Inline Title Editing ----

    _startTitleEdit(nodeId, titleEl) {
        if (titleEl.querySelector('input')) return;

        const node = this.nodes.get(nodeId);
        if (!node) return;

        const currentTitle = node.title;
        const input = document.createElement('input');
        input.type = 'text';
        input.className = 'ng-node-title-input';
        input.value = currentTitle;

        titleEl.textContent = '';
        titleEl.appendChild(input);
        input.focus();
        input.select();

        input.addEventListener('mousedown', e => e.stopPropagation());
        input.addEventListener('keydown', e => {
            if (e.key === 'Enter') {
                e.preventDefault();
                this._commitTitleEdit(nodeId, titleEl, input);
            } else if (e.key === 'Escape') {
                e.preventDefault();
                this._cancelTitleEdit(nodeId, titleEl, currentTitle);
            }
            e.stopPropagation();
        });
        input.addEventListener('blur', () => {
            this._commitTitleEdit(nodeId, titleEl, input);
        });
    }

    _commitTitleEdit(nodeId, titleEl, input) {
        if (!titleEl.contains(input)) return;
        this._pushUndoState();
        const newTitle = input.value.trim() || 'Step';
        const node = this.nodes.get(nodeId);
        if (node) node.title = newTitle;
        titleEl.textContent = newTitle;
        this._dispatchChange();
    }

    _cancelTitleEdit(nodeId, titleEl, originalTitle) {
        titleEl.textContent = originalTitle;
    }

    // ---- Undo / Redo ----

    _getSnapshot() {
        const nodes = [];
        for (const node of this.nodes.values()) {
            nodes.push({
                id: node.id,
                type: node.type,
                title: node.title,
                tool: node.tool,
                description: node.description,
                x: node.x,
                y: node.y,
                width: node.width,
                height: node.height,
                status: node.status,
                stepNum: node.stepNum,
                resultText: node.resultText,
                portValues: JSON.parse(JSON.stringify(node.portValues || {}))
            });
        }
        const connections = [];
        for (const conn of this.connections.values()) {
            connections.push({
                id: conn.id,
                from: conn.from,
                fromPort: conn.fromPort,
                to: conn.to,
                toPort: conn.toPort,
                type: conn.type
            });
        }
        return { nodes, connections, nextNodeId: this.nextNodeId, nextConnId: this.nextConnId };
    }

    _restoreSnapshot(snapshot) {
        if (!snapshot) return;
        this.nodes.clear();
        this.connections.clear();
        if (this.canvas) this.canvas.innerHTML = '';
        if (this.svg) this.svg.innerHTML = '';
        this.selectedNode = null;
        this.selectedConnection = null;
        this.selectedNodes.clear();

        this.nextNodeId = snapshot.nextNodeId || 1;
        this.nextConnId = snapshot.nextConnId || 1;

        for (const n of snapshot.nodes) {
            const node = {
                id: n.id,
                type: n.type,
                title: n.title,
                tool: n.tool || '',
                description: n.description || '',
                x: n.x,
                y: n.y,
                width: n.width,
                height: n.height,
                status: n.status || 'pending',
                stepNum: n.stepNum || '',
                resultText: n.resultText || '',
                portValues: JSON.parse(JSON.stringify(n.portValues || {}))
            };
            this.nodes.set(node.id, node);
            this._renderNode(node);
        }
        for (const c of snapshot.connections) {
            const conn = {
                id: c.id,
                from: c.from,
                fromPort: c.fromPort,
                to: c.to,
                toPort: c.toPort,
                type: c.type || 'flow'
            };
            this.connections.set(conn.id, conn);
            this._renderConnection(conn);
            if (conn.type === 'flow') {
                this._hideDefaultField(conn.to, conn.toPort);
            }
        }
        this._applyTransform();
        this.refreshConnections();
    }

    _pushUndoState() {
        this._undoStack.push(this._getSnapshot());
        this._redoStack = [];
        if (this._undoStack.length > this._maxHistory) {
            this._undoStack.shift();
        }
    }

    _undo() {
        if (this._undoStack.length === 0) return;
        this._redoStack.push(this._getSnapshot());
        const snapshot = this._undoStack.pop();
        this._restoreSnapshot(snapshot);
        this._dispatchChange();
    }

    _redo() {
        if (this._redoStack.length === 0) return;
        this._undoStack.push(this._getSnapshot());
        const snapshot = this._redoStack.pop();
        this._restoreSnapshot(snapshot);
        this._dispatchChange();
    }

    _dispatchChange() {
        this.container.dispatchEvent(new CustomEvent('graph-layout-changed'));
    }

    // ---- Utilities ----

    _escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }
}

// Export for use in standalone page
if (typeof module !== 'undefined' && module.exports) {
    module.exports = NodeGraph;
}
