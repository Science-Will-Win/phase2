// ============================================
// Observe Node Type - displays text results
// ============================================

NodeRegistry.register('observe', {
    label: 'Observe',
    category: 'General',
    result: true,

    ports: [
        { name: 'in', dir: 'in', type: 'any' }
    ],

    defaultConfig: {
        title: 'Observe',
        tool: '',
        description: '',
        status: 'pending',
        stepNum: '',
        resultText: ''
    },

    render(node, helpers) {
        const el = document.createElement('div');
        const result = node.resultText || '';
        const outputContent = result
            ? helpers.escapeHtml(result)
            : '<span class="ng-observe-placeholder">No result yet</span>';

        el.innerHTML = `
            <div class="ng-ports-in-row" data-node-id="${node.id}">
                <div class="ng-port ng-port-in" data-port-name="in" data-port-dir="in" data-port-type="any" data-node-id="${node.id}"></div>
            </div>
            <div class="ng-node-header">
                <span class="ng-node-num">${node.stepNum}</span>
                <span class="ng-node-title">${helpers.escapeHtml(node.title)}</span>
            </div>
            <div class="ng-node-body ng-observe-body">
                <div class="ng-observe-output">${outputContent}</div>
            </div>
        `;

        return el;
    },

    getDragHandle(el) {
        return el.querySelector('.ng-node-header');
    },

    updateResult(el, text) {
        const output = el.querySelector('.ng-observe-output');
        if (!output) return;
        if (text) {
            const div = document.createElement('div');
            div.textContent = text;
            output.innerHTML = div.innerHTML;
        } else {
            output.innerHTML = '<span class="ng-observe-placeholder">No result yet</span>';
        }
    }
});
