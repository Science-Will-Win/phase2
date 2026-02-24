// ============================================
// Save Node Type - saves input data to file
// ============================================

(function () {

function _resolveTags(template, nodeTitle) {
    const now = new Date();
    const pad = (n) => String(n).padStart(2, '0');
    const weekdays = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'];
    const weekdaysShort = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
    return template
        .replace(/\{date\}/g, `${now.getFullYear()}-${pad(now.getMonth()+1)}-${pad(now.getDate())}`)
        .replace(/\{date-short\}/g, `${pad(now.getMonth()+1)}-${pad(now.getDate())}`)
        .replace(/\{year\}/g, String(now.getFullYear()))
        .replace(/\{month\}/g, pad(now.getMonth()+1))
        .replace(/\{day\}/g, pad(now.getDate()))
        .replace(/\{time\}/g, `${pad(now.getHours())}-${pad(now.getMinutes())}-${pad(now.getSeconds())}`)
        .replace(/\{time-short\}/g, `${pad(now.getHours())}-${pad(now.getMinutes())}`)
        .replace(/\{hour\}/g, pad(now.getHours()))
        .replace(/\{minute\}/g, pad(now.getMinutes()))
        .replace(/\{second\}/g, pad(now.getSeconds()))
        .replace(/\{weekday\}/g, weekdays[now.getDay()])
        .replace(/\{weekday-short\}/g, weekdaysShort[now.getDay()])
        .replace(/\{timestamp\}/g, String(Math.floor(now.getTime() / 1000)))
        .replace(/\{uuid\}/g, (typeof crypto !== 'undefined' && crypto.randomUUID)
            ? crypto.randomUUID().slice(0, 8)
            : Math.random().toString(36).slice(2, 10))
        .replace(/\{node-title\}/g, nodeTitle || 'Save');
}

function _updatePreview(el, node) {
    const preview = el.querySelector('.ng-save-preview');
    if (!preview) return;
    const pv = node.portValues || {};
    const pathVal = pv.path || './output';
    const nameVal = pv.name || 'result_{date}_{time-short}';
    const resolved = _resolveTags(nameVal, node.title);
    preview.textContent = pathVal.replace(/\/+$/, '') + '/' + resolved;
}

NodeRegistry.register('save', {
    label: 'Save',
    category: 'General',
    result: true,

    ports: [
        { name: 'in', dir: 'in', type: 'any' }
    ],

    defaultConfig: {
        title: 'Save',
        status: 'pending',
        resultText: '',
        portValues: { path: './output', name: 'result_{date}_{time-short}' }
    },

    render(node, helpers) {
        const el = document.createElement('div');
        const pv = node.portValues || {};
        const pathVal = pv.path !== undefined ? pv.path : './output';
        const nameVal = pv.name !== undefined ? pv.name : 'result_{date}_{time-short}';
        const resolved = _resolveTags(nameVal, node.title);
        const previewPath = pathVal.replace(/\/+$/, '') + '/' + resolved;
        const result = node.resultText || '';

        el.innerHTML = `
            <div class="ng-ports-in-row" data-node-id="${node.id}">
                <div class="ng-port ng-port-in" data-port-name="in" data-port-dir="in" data-port-type="any" data-node-id="${node.id}"></div>
            </div>
            <div class="ng-node-header">
                <span class="ng-node-title">${helpers.escapeHtml(node.title)}</span>
            </div>
            <div class="ng-node-body ng-save-body">
                <div class="ng-save-row">
                    <span class="ng-save-label">Path</span>
                    <input class="ng-save-field ng-interactive" type="text"
                           value="${helpers.escapeHtml(pathVal)}" data-save-field="path"
                           placeholder="./output">
                </div>
                <div class="ng-save-row">
                    <span class="ng-save-label">Name</span>
                    <input class="ng-save-field ng-interactive" type="text"
                           value="${helpers.escapeHtml(nameVal)}" data-save-field="name"
                           placeholder="result_{date}_{time-short}">
                </div>
                <div class="ng-save-preview" title="Resolved preview">${helpers.escapeHtml(previewPath)}</div>
                <div class="ng-save-status">${result ? helpers.escapeHtml(result) : ''}</div>
            </div>
        `;

        return el;
    },

    afterRender(el, node, helpers) {
        el.querySelectorAll('.ng-save-field').forEach(input => {
            const update = () => {
                const n = helpers.getNode();
                if (!n) return;
                if (!n.portValues) n.portValues = {};
                const field = input.dataset.saveField;
                n.portValues[field] = input.value;
                _updatePreview(el, n);
            };
            input.addEventListener('change', update);
            input.addEventListener('input', update);
        });
    },

    getDragHandle(el) {
        return el.querySelector('.ng-node-header');
    },

    updateResult(el, text) {
        const status = el.querySelector('.ng-save-status');
        if (!status) return;
        status.textContent = text || '';
    }
});

})();
