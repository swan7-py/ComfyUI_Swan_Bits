// Dynamic colour inputs for Swan SAM3D Track Palette.
// Same shape as KJNodes' *Multi nodes: an inputcount widget plus an "Update inputs"
// button rebuilds color_1..color_N. Slots are linkable STRING inputs, so a linked
// slot stores its link by name and growing the list never shifts earlier values.
const { app } = window.comfyAPI.app;

const PREFIX = "color_";

function setupDynamicInputs(node) {
    const rebuild = () => {
        if (!node.inputs) node.inputs = [];
        const countWidget = node.widgets?.find((w) => w.name === "inputcount");
        if (!countWidget) return;
        const target = Math.max(1, Number(countWidget.value) || 1);
        const current = node.inputs.filter((i) => i.name?.startsWith(PREFIX)).length;
        if (target === current) return;
        if (target < current) {
            for (let i = 0; i < current - target; i++) node.removeInput(node.inputs.length - 1);
        } else {
            for (let i = current + 1; i <= target; i++) node.addInput(`${PREFIX}${i}`, "STRING");
        }
        node.setSize([node.size[0], node.computeSize()[1]]);
    };

    node.addWidget("button", "Update inputs", null, rebuild);

    // Restore slots when a saved graph loads: the widget value is set without a
    // canvas argument, which is the tell between a reload and an interactive drag.
    const countWidget = node.widgets?.find((w) => w.name === "inputcount");
    if (countWidget) {
        const original = countWidget.callback;
        countWidget.callback = function (value, canvas) {
            const result = original ? original.apply(this, arguments) : undefined;
            if (!canvas) rebuild();
            return result;
        };
    }
    return rebuild;
}

app.registerExtension({
    name: "SwanBits.SAM3DTrackPalette",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "SWAN_SAM3DTrackPalette") return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            setupDynamicInputs(this)();
            return result;
        };
    },
});
