// The compiler scene's data against its own invariants: what scripts/pipeline-excerpts.mjs writes has to
// be drawable as it stands, a DAG whose layout respects its edges, buffers that fit the strip they
// pack into, and stages that agree with each other where they describe the same kernel.

import { readFileSync } from "node:fs"
import { join } from "node:path"
import { describe, expect, it } from "vitest"
import data from "../src/data/pipeline/gemm_fp32.json"
import { countAstNodes, liveBuffers, recommendedPasses, type PipelineData } from "../src/components/home/pipeline-data"

const pipeline = data as unknown as PipelineData
const { source, ast, graph, passes, mlir, ptx, gpu } = pipeline.structure
const REPO = join(import.meta.dirname, "../..")

describe("stages", () => {
  it("keeps a text excerpt for every station the scene shows", () => {
    for (const id of ["source", "ast", "graph", "passes", "mlir", "ptx"]) {
      expect(typeof pipeline.stages[id]).toBe("string")
      expect(pipeline.stages[id].length).toBeGreaterThan(0)
    }
  })
})

describe("source", () => {
  it("is the file as it is now", () => {
    const lines = readFileSync(join(REPO, source.path), "utf8").replace(/\r\n?/g, "\n").split("\n")
    for (const line of source.lines) expect(line.text).toBe(lines[line.n - 1].replace(/\s+$/, ""))
  })

  it("marks the loop's header, body and end", () => {
    const role = (n: number) => source.lines.find(line => line.n === n)?.role
    expect(role(source.loop.header)).toBe("loop-header")
    expect(source.loop.body.length).toBeGreaterThan(0)
    for (const n of source.loop.body) {
      expect(n).toBeGreaterThan(source.loop.header)
      expect(n).toBeLessThan(source.loop.end)
      expect(role(n)).toBe("loop-body")
    }
    expect(role(source.loop.end)).toBe("loop-end")
  })
})

describe("ast", () => {
  it("is pruned to a size that reads", () => {
    expect(countAstNodes(ast.root)).toBe(ast.nodeCount)
    expect(ast.nodeCount).toBeGreaterThanOrEqual(25)
    expect(ast.nodeCount).toBeLessThanOrEqual(40)
  })

  it("keeps the loop whole, at the line the source marks", () => {
    const loop = ast.root.children?.find(child => child.kind === ast.focus)
    expect(loop?.line).toBe(source.loop.header)
    expect(loop?.folded).toBeUndefined()
    const body = loop?.children?.filter(child => child.role === "body").map(child => child.line)
    expect(body).toEqual(source.loop.body)
  })
})

describe("graph", () => {
  const ids = new Set(graph.nodes.map(node => node.id))
  const byId = new Map(graph.nodes.map(node => [node.id, node]))

  it("has unique node ids and edges between existing nodes", () => {
    expect(ids.size).toBe(graph.nodes.length)
    for (const edge of graph.edges) {
      expect(ids.has(edge.from), edge.from).toBe(true)
      expect(ids.has(edge.to), edge.to).toBe(true)
    }
  })

  it("reads on a front page", () => {
    expect(graph.nodes.length).toBeLessThanOrEqual(40)
    expect(graph.pruned.foldedOps).toBe(graph.pruned.foldedIds.length)
  })

  it("has finite coordinates on the layer and row grid", () => {
    for (const node of graph.nodes) {
      expect(Number.isFinite(node.x) && Number.isFinite(node.y), node.id).toBe(true)
      expect(node.x).toBe(node.layer * graph.layout.columnWidth)
      expect(node.y).toBe(node.row * graph.layout.rowHeight)
      expect(node.layer).toBeLessThan(graph.layout.layers)
      expect(node.row).toBeLessThan(graph.layout.rows)
      expect(node.x).toBeLessThanOrEqual(graph.layout.width)
      expect(node.y).toBeLessThanOrEqual(graph.layout.height)
    }
  })

  it("puts no two nodes in one cell", () => {
    const cells = graph.nodes.map(node => `${node.layer},${node.row}`)
    expect(new Set(cells).size).toBe(cells.length)
  })

  it("layers every forward edge left to right", () => {
    for (const edge of graph.edges.filter(edge => edge.kind === "data")) {
      expect(byId.get(edge.from)!.layer, `${edge.from} -> ${edge.to}`).toBeLessThan(byId.get(edge.to)!.layer)
    }
  })

  it("closes the loop with one back edge, from its yield to its carried argument", () => {
    const back = graph.edges.filter(edge => edge.kind === "back")
    expect(back).toHaveLength(1)
    expect(byId.get(back[0].from)!.kind).toBe("yield")
    expect(byId.get(back[0].to)!.kind).toBe("block_arg")
  })

  it("boxes the loop body and nothing else", () => {
    for (const region of graph.regions) {
      const { box } = region
      const inside = (node: { x: number; y: number }) =>
        node.x > box.x && node.x < box.x + box.width && node.y > box.y && node.y < box.y + box.height
      const members = graph.nodes.filter(node => node.region === region.id)
      expect(members.length).toBeGreaterThan(0)
      for (const node of graph.nodes) expect(inside(node), node.id).toBe(node.region === region.id)
      const header = byId.get(region.header)!
      const result = byId.get(region.result)!
      expect(header.kind).toBe("for")
      expect(result.kind).toBe("for_result")
      expect(header.layer).toBe(box.firstLayer - 1)
      expect(result.layer).toBe(box.lastLayer + 1)
    }
  })
})

describe("passes", () => {
  it("recommends exactly one", () => {
    expect(passes.filter(view => view.recommended)).toHaveLength(1)
  })

  for (const view of passes) {
    describe(view.trace, () => {
      const opIds = new Set(view.ops.map(op => op.id))

      it("is a trace the visualizer has", () => {
        const trace = JSON.parse(readFileSync(join(REPO, `website/src/data/traces/${view.trace}.json`), "utf8"))
        expect(trace.sharedPeakBytes).toBe(view.peakBytes)
        expect(trace.buffers).toHaveLength(view.buffers.length)
      })

      it("fits every buffer within the peak, live over ops that exist", () => {
        for (const buffer of view.buffers) {
          expect(buffer.offsetBytes).toBeGreaterThanOrEqual(0)
          expect(buffer.bytes).toBeGreaterThan(0)
          expect(buffer.offsetBytes + buffer.bytes).toBeLessThanOrEqual(view.peakBytes)
          expect(opIds.has(buffer.firstOp) && opIds.has(buffer.lastOp)).toBe(true)
          expect(buffer.firstOp).toBeLessThanOrEqual(buffer.lastOp)
        }
        expect(Math.max(...view.buffers.map(buffer => buffer.offsetBytes + buffer.bytes))).toBe(view.peakBytes)
      })

      it("places barriers after ops that exist, and counts them", () => {
        for (const barrier of view.barriers) expect(opIds.has(barrier.afterOp)).toBe(true)
        const kept = view.barriers.filter(barrier => barrier.kept).length
        expect(view.barrierCounts).toEqual({ kept, elided: view.barriers.length - kept })
      })

      it("orders ops by id, a loop after its body", () => {
        const ids = view.ops.map(op => op.id)
        expect(ids).toEqual([...ids].sort((a, b) => a - b))
        for (const loop of view.loops) {
          expect(loop.firstOp).toBeLessThan(loop.lastOp)
          expect(view.ops.filter(op => op.parent === loop.id).every(op => op.id < loop.id)).toBe(true)
        }
      })
    })
  }

  it("shows the strip being packed: buffers come and go over the timeline", () => {
    const view = recommendedPasses(pipeline.structure)
    const counts = view.ops.map(op => liveBuffers(view, op.id).length)
    expect(Math.max(...counts)).toBeGreaterThan(1)
    expect(new Set(view.buffers.map(buffer => buffer.offsetBytes)).size).toBeLessThan(view.buffers.length)
  })
})

describe("mlir", () => {
  it("counts every op once", () => {
    const sum = (record: Record<string, number>) => Object.values(record).reduce((a, b) => a + b, 0)
    expect(sum(mlir.dialects)).toBe(mlir.ops)
    type Region = typeof mlir.root
    const total = (region: Region): number =>
      sum(region.ops) + region.children.reduce((s, child) => s + 1 + total(child) - (child.op === "else" ? 1 : 0), 0)
    expect(total(mlir.root)).toBe(mlir.root.total)
  })

  it("has the main loop carrying the accumulator, and at most three labels", () => {
    expect(mlir.loops.some(loop => loop.depth === 0 && loop.iterArgs)).toBe(true)
    expect(mlir.labels.length).toBeGreaterThan(0)
    expect(mlir.labels.length).toBeLessThanOrEqual(3)
  })
})

describe("ptx", () => {
  it("files every instruction under one family", () => {
    const total = ptx.families.reduce((sum, family) => sum + family.count, 0)
    expect(total).toBe(ptx.instructions)
    expect(Object.values(ptx.opcodes).reduce((a, b) => a + b, 0)).toBe(ptx.instructions)
    for (const name of ["ld.global", "ld.shared", "st.shared", "fma", "bar.sync"]) {
      expect(ptx.families.find(family => family.name === name)?.count, name).toBeGreaterThan(0)
    }
  })
})

describe("across stages", () => {
  it("agrees on the launch and the shared memory", () => {
    expect(ptx.maxThreads).toBe(gpu.threads)
    expect(gpu.warps * gpu.lanes).toBe(gpu.threads)
    expect(gpu.ctas).toBe(gpu.grid[0] * gpu.grid[1] * gpu.grid[2])
    expect(ptx.sharedBytes).toBe(gpu.sharedBytesPerCta)
    expect(mlir.sharedBytes).toBe(gpu.sharedBytesPerCta)
    const asWritten = passes.find(view => view.trace === gpu.trace)
    expect(asWritten?.peakBytes).toBe(gpu.sharedBytesPerCta)
  })
})
