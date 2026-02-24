// ============================================
// Visualize Node Type - displays images/charts
// ============================================

NodeRegistry.register('visualize', {
    label: 'Visualize',
    category: 'General',
    result: true,

    ports: [
        { name: 'in', dir: 'in', type: 'any' },
        { name: 'out', dir: 'out', type: 'any' }
    ],

    defaultConfig: {
        title: 'Visualize',
        tool: '',
        description: '',
        status: 'pending',
        stepNum: '',
        resultText: ''
    },

    render(node, helpers) {
        const el = document.createElement('div');
        const hasResult = node.resultText && node.resultText.length > 0;
        const bodyStyle = hasResult ? '' : ' style="display:none"';

        el.innerHTML = `
            <div class="ng-ports-in-row" data-node-id="${node.id}">
                <div class="ng-port ng-port-in" data-port-name="in" data-port-dir="in" data-port-type="any" data-node-id="${node.id}"></div>
            </div>
            <div class="ng-node-header">
                <span class="ng-node-num">${node.stepNum}</span>
                <span class="ng-node-title">${helpers.escapeHtml(node.title)}</span>
            </div>
            <div class="ng-node-body ng-visualize-body"${bodyStyle}>
                <div class="ng-visualize-output"></div>
            </div>
            <div class="ng-node-progress"></div>
            <div class="ng-ports-out-row" data-node-id="${node.id}">
                <div class="ng-port ng-port-out" data-port-name="out" data-port-dir="out" data-port-type="any" data-node-id="${node.id}"></div>
            </div>
        `;

        if (hasResult) {
            const output = el.querySelector('.ng-visualize-output');
            this._renderResult(output, node.resultText);
        }

        return el;
    },

    getDragHandle(el) {
        return el.querySelector('.ng-node-header');
    },

    _renderResult(container, resultData) {
        if (!container || !resultData) return;
        if (resultData.startsWith('data:image') || resultData.startsWith('http')) {
            container.innerHTML = `<img src="${resultData}" alt="Visualization">`;
        } else {
            container.textContent = resultData;
        }
    },

    updateResult(el, resultData) {
        const body = el.querySelector('.ng-visualize-body');
        const output = el.querySelector('.ng-visualize-output');
        if (!body || !output) return;

        if (resultData) {
            body.style.display = '';
            this._renderResult(output, resultData);
        } else {
            body.style.display = 'none';
            output.innerHTML = '';
        }
    }
});
