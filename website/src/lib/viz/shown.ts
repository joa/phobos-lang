// What a code panel shows of a kernel's source: every line but `@cluster` ones, with the original
// line number of each shown line, so trace spans still land on the right line. Pure.

export interface ShownSource {
  code: string
  lineMap: number[]
  hiddenLines: number[]
}

export function shownSource(text: string): ShownSource {
  const source = text.replace(/\r\n?/g, "\n").replace(/\n+$/, "")
  const hiddenLines: number[] = []
  const lineMap: number[] = []
  const shown: string[] = []
  source.split("\n").forEach((line, index) => {
    if (/^\s*@cluster\b/.test(line)) hiddenLines.push(index + 1)
    else {
      shown.push(line)
      lineMap.push(index + 1)
    }
  })
  return { code: shown.join("\n"), lineMap, hiddenLines }
}
