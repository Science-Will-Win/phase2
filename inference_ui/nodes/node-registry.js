// ============================================
// Port Type System
// ============================================

const PortTypes = {
    _groups: {},
    _memberOf: {},

    defineGroup(groupName) {
        if (!this._groups[groupName]) this._groups[groupName] = new Set();
    },

    define(typeName, memberOfGroups = []) {
        this._memberOf[typeName] = new Set(memberOfGroups);
        for (const g of memberOfGroups) {
            if (!this._groups[g]) this._groups[g] = new Set();
            this._groups[g].add(typeName);
        }
    },

    isGroup(typeName) {
        return this._groups.hasOwnProperty(typeName);
    },

    isCompatible(outType, inType) {
        if (outType === 'image') return inType === 'image';
        if (inType === 'image') return outType === 'image';
        if (inType === 'any' || outType === 'any') return true;

        if (this.isGroup(inType)) {
            const accepted = this._groups[inType];
            if (accepted.has(outType)) return true;
            if (this.isGroup(outType)) {
                const outMembers = this._groups[outType];
                if (outMembers.size === 0) return false;
                for (const m of outMembers) {
                    if (!accepted.has(m)) return false;
                }
                return true;
            }
            return false;
        }

        if (inType === 'table') return outType === 'table' || outType === 'data';
        if (inType === 'string') return true;
        if (inType === 'float' && outType === 'int') return true;
        return outType === inType;
    }
};

PortTypes.defineGroup('numeric');
PortTypes.defineGroup('addable');

PortTypes.define('float',   ['numeric', 'addable']);
PortTypes.define('int',     ['numeric', 'addable']);
PortTypes.define('double',  ['numeric', 'addable']);
PortTypes.define('string',  ['addable']);
PortTypes.define('matrix',  ['numeric', 'addable']);
PortTypes.define('vector2', ['numeric', 'addable']);
PortTypes.define('vector3', ['numeric', 'addable']);
PortTypes.define('vector4', ['numeric', 'addable']);
PortTypes.define('color',   ['numeric', 'addable']);
PortTypes.define('boolean', []);
PortTypes.define('data',    []);
PortTypes.define('table',   []);
PortTypes.define('image',   []);

// ============================================
// Node Type Registry
// ============================================

const NodeRegistry = {
    _types: {},

    register(type, definition) {
        if (!type || !definition) return;
        if (!definition.render || typeof definition.render !== 'function') {
            console.warn(`NodeRegistry: type "${type}" missing render(). Skipped.`);
            return;
        }
        definition.category = definition.category || 'General';
        definition.label = definition.label || type;
        definition.ports = definition.ports || [
            { name: 'in', dir: 'in', type: 'any' },
            { name: 'out', dir: 'out', type: 'any' }
        ];
        this._types[type] = definition;
    },

    get(type) {
        return this._types[type] || this._types['step'] || null;
    },

    getAll() {
        return Object.keys(this._types);
    },

    getByCategory() {
        const result = {};
        for (const [type, def] of Object.entries(this._types)) {
            const cat = def.category || 'General';
            if (!result[cat]) result[cat] = [];
            result[cat].push({ type, label: def.label || type, definition: def });
        }
        return result;
    },

    async loadAll() {
        const resp = await fetch('/api/node-manifest');
        if (!resp.ok) {
            console.error('NodeRegistry: failed to fetch node manifest');
            return;
        }
        const { files } = await resp.json();
        const load = (src) => new Promise((resolve) => {
            if (document.querySelector(`script[src="${src}"]`)) { resolve(); return; }
            const s = document.createElement('script');
            s.src = src;
            s.onload = resolve;
            s.onerror = () => { console.warn(`NodeRegistry: failed to load ${src}`); resolve(); };
            document.head.appendChild(s);
        });
        const helpers = files.filter(f => f.includes('-helper') || f.includes('-util'));
        const nodes = files.filter(f => !f.includes('-helper') && !f.includes('-util'));
        for (const src of helpers) await load(src);
        await Promise.all(nodes.map(load));
    }
};
