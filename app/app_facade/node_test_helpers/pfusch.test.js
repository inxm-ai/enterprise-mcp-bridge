import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { pfuschTest, setupDomStubs } from "./domstubs.js";

setupDomStubs();
const { html, pfusch, script } = await import("./pfusch.js");

globalThis.pfuschLifecycle = { cleanups: 0, runs: 0 };

pfusch("runtime-probe", { items: [] }, (state) => [
    script(function () {
        globalThis.pfuschLifecycle.runs += 1;
        return () => {
            globalThis.pfuschLifecycle.cleanups += 1;
        };
    }),
    html.output(String(state.items.length)),
    0,
]);

pfusch("lazy-probe", {}, () => [html.p("ready")]);

describe("current pfusch runtime", () => {
    it("supports the current state, lifecycle, rendering, and form contracts", async () => {
        const component = pfuschTest("runtime-probe");
        await component.flush();
        assert.equal(component.get("output").textContent, "0");
        assert.equal(component.get("span").textContent, "0");
        assert.equal(component.host.internals.formValue, undefined);

        component.host.state.mutate((state) =>
            state.items.push({ id: "one" }),
        );
        await component.flush();
        assert.equal(component.get("output").textContent, "1");

        const runsBeforeReconnect = globalThis.pfuschLifecycle.runs;
        component.host.remove();
        await component.flush();
        assert.equal(globalThis.pfuschLifecycle.cleanups, 1);
        document.body.appendChild(component.host);
        await component.flush();
        assert.equal(
            globalThis.pfuschLifecycle.runs,
            runsBeforeReconnect + 1,
        );

        const named = pfuschTest("runtime-probe", { name: "probe" });
        await named.flush();
        assert.equal(named.host.internals.formValue, '{"items":[]}');

        const lazy = pfuschTest("lazy-probe", { as: "lazy" });
        await lazy.flush();
        assert.equal(lazy.get("p").length, 0);
        lazy.host.removeAttribute("as");
        await lazy.flush();
        assert.equal(lazy.get("p").textContent, "ready");
    });
});
