// The latest release's tag, read once per build. The site deploys on every
// published release, so a build-time read stays current. A build that cannot
// reach GitHub renders without it rather than failing.

const LATEST_URL = "https://api.github.com/repos/joa/phobos-lang/releases/latest"

async function fetchLatestTag(): Promise<string | undefined> {
  const headers: Record<string, string> = { Accept: "application/vnd.github+json" }
  // CI runners share IPs, and the unauthenticated limit is per IP.
  const token = process.env.GITHUB_TOKEN
  if (token) headers.Authorization = `Bearer ${token}`
  try {
    const response = await fetch(LATEST_URL, { headers, signal: AbortSignal.timeout(10_000) })
    if (!response.ok) return undefined
    const { tag_name } = (await response.json()) as { tag_name?: unknown }
    return typeof tag_name === "string" ? tag_name : undefined
  } catch {
    return undefined
  }
}

let latest: Promise<string | undefined> | undefined

export function latestReleaseTag(): Promise<string | undefined> {
  latest ??= fetchLatestTag()
  return latest
}

const DOWNLOAD_URL = "https://github.com/joa/phobos-lang/releases/download"

export interface InstallCommand {
  os: "Linux" | "Windows"
  lang: "sh" | "powershell"
  command: string
}

/** One line per platform that downloads `tag` and unpacks it into a folder named after it. */
export function installCommands(tag: string): InstallCommand[] {
  const linux = `phobos-${tag}-linux-x64.tar.gz`
  const windows = `phobos-${tag}-windows-x64.zip`
  return [
    {
      os: "Linux",
      lang: "sh",
      command: `curl -fsSLO ${DOWNLOAD_URL}/${tag}/${linux} && tar -xzf ${linux} && rm ${linux}`,
    },
    {
      os: "Windows",
      lang: "powershell",
      command: `$ProgressPreference = 'SilentlyContinue'; iwr ${DOWNLOAD_URL}/${tag}/${windows} -OutFile ${windows}; Expand-Archive ${windows} .; rm ${windows}`,
    },
  ]
}
