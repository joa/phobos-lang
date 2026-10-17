// The shipped library's sources, bundled as text from phobos-lang, in the order stdlib.rs embeds
// them. A file stdlib.rs lists but this folder lacks, or the other way round, fails here.

import stdlibRs from "../../../../phobos-lang/src/stdlib.rs?raw"
import { parseLibrary, shippedFiles, type LibraryFile, type LibraryFunction } from "./library"

const TEXTS = import.meta.glob("../../../../phobos-lang/src/stdlib/*.ph", {
  query: "?raw",
  import: "default",
  eager: true,
}) as Record<string, string>

export const LIBRARY: LibraryFile[] = shippedFiles(stdlibRs).map(({ file, path }) => {
  const key = `../../../../phobos-lang/src/${path}`
  const text = TEXTS[key]
  if (text === undefined) throw new Error(`stdlib.rs embeds ${path}, which is not in phobos-lang/src/stdlib`)
  return parseLibrary(file, text)
})

export const FUNCTIONS: LibraryFunction[] = LIBRARY.flatMap(file => file.functions)

export function libraryFunction(name: string): LibraryFunction {
  const fn = FUNCTIONS.find(f => f.name === name)
  if (!fn) throw new Error(`the standard library has no function \`${name}\``)
  return fn
}
