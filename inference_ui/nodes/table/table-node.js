// ============================================
// Table Node Type - displays tabular data
// ============================================

NodeRegistry.register('table', {
    label: 'Table',
    category: 'General',
    result: true,

    ports: [
        { name: 'in', dir: 'in', type: 'table' },
        { name: 'out', dir: 'out', type: 'table' }
    ],

    defaultConfig: {
        title: 'Table',
        tool: '',
        description: '',
        status: 'pending',
        stepNum: '',
        resultText: ''
    },

    render(node, helpers) {
        const el = document.createElement('div');
        const hasResult = node.resultText && node.resultText.length > 0;

        el.innerHTML = `
            <div class="ng-ports-in-row" data-node-id="${node.id}">
                <div class="ng-port ng-port-in" data-port-name="in" data-port-dir="in" data-port-type="table" data-node-id="${node.id}"></div>
            </div>
            <div class="ng-node-header">
                <span class="ng-node-num">${node.stepNum}</span>
                <span class="ng-node-title">${helpers.escapeHtml(node.title)}</span>
            </div>
            <div class="ng-node-body ng-table-body">
                <div class="ng-table-output">
                    <span class="ng-table-placeholder">No data</span>
                </div>
            </div>
            <div class="ng-node-progress"></div>
            <div class="ng-ports-out-row" data-node-id="${node.id}">
                <div class="ng-port ng-port-out" data-port-name="out" data-port-dir="out" data-port-type="table" data-node-id="${node.id}"></div>
            </div>
        `;

        if (hasResult) {
            const output = el.querySelector('.ng-table-output');
            this._renderTableData(output, node.resultText);
        }

        return el;
    },

    getDragHandle(el) {
        return el.querySelector('.ng-node-header');
    },

    _renderTableData(container, data) {
        if (!container || !data) return;

        let rows;
        if (typeof data === 'string') {
            try { rows = JSON.parse(data); } catch { rows = null; }
        } else if (Array.isArray(data)) {
            rows = data;
        }

        if (!Array.isArray(rows) || rows.length === 0) {
            container.textContent = typeof data === 'string' ? data : JSON.stringify(data);
            return;
        }

        const escapeCell = v => {
            const d = document.createElement('span');
            d.textContent = v == null ? '' : String(v);
            return d.innerHTML;
        };

        const headerRow = rows[0];
        let html = '<table><thead><tr>';
        for (const h of headerRow) html += `<th>${escapeCell(h)}</th>`;
        html += '</tr></thead><tbody>';
        for (let r = 1; r < rows.length; r++) {
            html += '<tr>';
            for (const c of rows[r]) html += `<td>${escapeCell(c)}</td>`;
            html += '</tr>';
        }
        html += '</tbody></table>';
        container.innerHTML = html;
    },

    updateResult(el, data) {
        const output = el.querySelector('.ng-table-output');
        if (!output) return;
        if (data) {
            this._renderTableData(output, data);
        } else {
            output.innerHTML = '<span class="ng-table-placeholder">No data</span>';
        }
    }
});
