# Submission Discipline
- Don't hesitate to use sub-agents. Give them relevant instructions so they can do their task.
- Profile after major changes or major score gains.
- Use an advisor model when you are stuck or need fresh ideas. It can help break tunnel vision.
- Don't be afraid to implement hard tasks.
- Don't hesitate to take risks.
- Be open to new ideas and search the internet at times for new idea exposure.
- Before submitting, run the cheapest available sanity check for the candidate file.
- Treat a timeout as inconclusive, not as a correctness/performance rejection.
- Treat a completed residual failure or timing regression as real evidence.
- Only completed pass/fail/timing output is evidence.

# Beam Search Discipline
Do not optimize as a single-incumbent hill climb. Maintain a small beam of active idea families so that local negative results do not prematurely kill useful ingredients.

- Keep a live beam note in autoresearch/beams/.
- Each beam entry should record the parent candidate, hypothesis, exact changed functions or gates, current best candidate/log, and next singleton, combination, or kill decision.
- Keep at least 3 active beams when there is enough work:
  - one exploit beam near the current best,
  - one near-miss beam,
  - one structural/high-risk beam from profiling or external ideas,
  - one cleanup/compile-time beam only if it has not consumed the whole search.
- Use sub-agents to work different beams, not many variants of the same tiny parameter unless explicitly asked for a parallel sweep.
- Do not declare a family dead after isolated singletons fail.
- If two ideas are individually neutral or slightly slower but touch independent costs, try combining them before retiring the family unless correctness risk is high.
- Preserve near-misses as beam material when they show a repeatable isolated win, improve one important case, remove overhead, or change an algorithmic surface that can combine with another beam.
- Kill a beam only with a clear reason: inherent correctness failure, repeated meaningful regression after a reasonable retune, singleton and plausible combinations both lose, profile evidence shows the targeted cost is no longer material, or implementation cost is blocking higher-value beams.
- After every 3-5 submissions, update the beam note with the current beam ranking and next combination candidates.

# Promotion / Git Hygiene
- When a new candidate is promoted into the active file, stage the promoted candidate, active file, evidence autoresearch/beams, and any archive moves.
- Commit after promotion unless the user says not to.
- Use a lower-case commit subject and describe both what changed and what made the improvement in the commit body.

# Profiling / Evidence Habits
- Prefer current active evidence over older sidecar timings.
- Keep raw profile logs and JSON under autoresearch/beams.
- When a candidate is rejected or promoted, update the relevant doc in autoresearch/beams.
- Use rg for searching when possible.
- Archive older candidates, profile logs, and probe scripts so the root stays navigable.

# Known Tried Ideas
- Do not mirror every attempt summary here. Check autoresearch/beams before repeating an idea, and update the relevant doc when an idea is rejected or promoted.
- Do not repeat rejected ideas as-is. If revisiting one, make the algorithmic difference explicit in the autoresearch/beams.
- If a platform or evaluator rule rejects a class of ideas, record it clearly so future agents do not rediscover the same invalid path.