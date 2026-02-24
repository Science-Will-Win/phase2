// ============================================
// Add Node Type (Math category)
// ============================================

NodeRegistry.register('math_add', {
    label: 'Add',
    category: 'Math',

    ports: [
        { name: 'a', dir: 'in', type: 'addable', label: 'A', defaultValue: 0 },
        { name: 'b', dir: 'in', type: 'addable', label: 'B', defaultValue: 0 },
        { name: 'out', dir: 'out', type: 'addable', label: 'Result' }
    ],

    resolveOutputType(inputTypes) {
        if (inputTypes.length === 0) return 'addable';
        if (inputTypes.some(t => t === 'string')) return 'string';
        const allNumeric = inputTypes.every(t =>
            PortTypes._groups['numeric']?.has(t) || t === 'numeric'
        );
        if (allNumeric) return 'numeric';
        return 'addable';
    },

    defaultConfig: {
        title: 'Add',
        status: 'pending',
        portValues: { a: 0, b: 0 }
    },

    render(node, helpers) {
        return MathNodeHelper.render(node, helpers, [
            { name: 'a', label: 'A', type: 'addable', defaultValue: 0 },
            { name: 'b', label: 'B', type: 'addable', defaultValue: 0 }
        ], 'addable');
    },

    getDragHandle(el) {
        return el.querySelector('.ng-node-header');
    }
});
