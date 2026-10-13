// Trace fences: in an MDX page, a whole-kernel fence with a `trace` meta,
//
//   ```phobos trace="lanes_matvec" dims="K=256, N=32" grid="8"
//
// becomes the same <Kernel> a kernel file gets, with that trace, so the visualizer attaches around it.
// The fence stays a plain code block when no trace of that name is committed yet, so a page can carry
// a fence before its trace is generated.
//
// Astro 7 renders Markdown with Satteri, which runs mdast plugins (the remark AST, visited from
// Rust) rather than remark's, so this is an mdast plugin, registered by the integration below at
// the front of the processor's list, ahead of the code highlighter.

import { existsSync, readdirSync } from "node:fs"
import { fileURLToPath } from "node:url"

const KERNEL_PATH = fileURLToPath(new URL("../src/components/Kernel.astro", import.meta.url)).replace(/\\/g, "/")
const TRACES_DIR = fileURLToPath(new URL("../src/data/traces/", import.meta.url))
const LOCAL_NAME = "PhobosTraceKernel"

/** The `trace` name in a fence's meta string, or null. */
export function traceOf(meta) {
  const match = /(?:^|\s)trace="([^"]+)"/.exec(meta ?? "")
  return match ? match[1] : null
}

/** Whether a trace of this base name is committed, for any chip. */
export function hasTrace(name, dir = TRACES_DIR) {
  if (!existsSync(dir)) return false
  return readdirSync(dir).some(file => file.startsWith(`${name}.`) && file.endsWith(".json"))
}

/** The MDX that replaces a fence: the fence body exactly, as a string literal, and the trace. */
export function kernelMdx(code, trace) {
  return `<${LOCAL_NAME} code={${JSON.stringify(code)}} trace=${JSON.stringify(trace)} />`
}

export function importMdx() {
  return `import ${LOCAL_NAME} from ${JSON.stringify(KERNEL_PATH)}`
}

/** The mdast plugin. `hasTraceNamed` is injectable for tests. */
export function traceFencesPlugin({ hasTraceNamed = name => hasTrace(name) } = {}) {
  const replacedIn = new WeakSet()
  return {
    name: "phobos-trace-fences",
    code(node, ctx) {
      if (ctx.sourceFormat !== "mdx" || node.lang !== "phobos") return
      const trace = traceOf(node.meta)
      if (!trace || !hasTraceNamed(trace)) return
      replacedIn.add(ctx.data)
      return { raw: kernelMdx(node.value, trace) }
    },
    after(root, ctx) {
      if (replacedIn.has(ctx.data)) ctx.prependChild(root, { type: "mdxjsEsm", value: importMdx() })
    },
  }
}

/** Registers the plugin with Astro's Satteri processor. */
export default function traceFences() {
  return {
    name: "phobos-trace-fences",
    hooks: {
      "astro:config:setup": ({ config, logger }) => {
        const processor = config.markdown?.processor
        if (processor?.name === "satteri" && Array.isArray(processor.options?.mdastPlugins)) {
          processor.options.mdastPlugins.unshift(traceFencesPlugin())
        } else {
          logger.warn("trace fences need the Satteri Markdown processor; fences stay plain code blocks")
        }
      },
    },
  }
}
