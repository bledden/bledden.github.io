---
title: 'A Memory Tool for 1% of Your Budget: How Recall Works, and Why I Rely on It'
description: 'A local, cross-session memory plugin for Claude Code and Cowork. How it works from the system up, and the measured answer to what it costs: about 1% of my token budget across 470,000 requests on three machines.'
pubDate: 2026-07-05
heroImage: '../../assets/recall-hero.png'
---

<style>
figure.chart{margin:2.5em 0;padding:1.9em 1.9em 1.5em;background:#fff;border:1px solid rgb(var(--gray-light));border-radius:10px;box-shadow:0 2px 10px rgba(var(--gray),14%);}
figure.chart .chart-title{font-weight:700;color:rgb(var(--black));font-size:1.05em;margin:0 0 1.3em;line-height:1.3;}
figure.chart svg,figure.chart img{display:block;width:100%;height:auto;}
figure.chart img{border-radius:6px;}
figure.chart figcaption{margin-top:1.4em;color:rgb(var(--gray));font-size:0.85em;line-height:1.55;}
</style>

---

A good portion of my week as an AI Engineer involves long-running Claude Code sessions that can stretch across weeks and compact many times over. Around it run a few personal projects: GPU kernels in Triton, a Metal backend for one of them, a quantum surface-code decoder, and the RunPod infrastructure they run on. In my professional world, I have leveraged long-running Claude sessions for QA pipelines, feature work, kernel development, and model experimentation. All of it lives across different machines and repos, and every compaction drops the specifics I was working with to keep the session under the window. The full *working set* is far larger than any single context can hold.

So I built **`/recall:recall`**, a Claude Code (and now Claude Cowork) plugin. It captures every exchange to a local SQLite database and hands the relevant part back when I ask for it. The core idea is simple. **Claude should never truly lose context**, not after a compaction, not weeks later, not in a different session. The only cost is some tokens when I pull that context back in. Five months on, it is the tool I reach for most, and in my heaviest sessions it accounts for *nearly all* of my skill usage.

---

## Where context goes

This is not something unique to me. Anyone working in long sessions runs into it. The useful context keeps growing while the space to hold it does not. Lose a session outright, to a crash or a stray `/clear`, and you are left with only what you thought to write down. Two things make it worse when the work is broad:

- **Breadth.** Several projects run at once, each with its own commands, decisions, and dead-ends, spread across separate sessions and machines. The detail I need was often settled in a *different* conversation days ago.
- **Depth.** A single thread can run for a week, and Claude Code compacts it to fit, keeping the summary and losing the detail you were in the middle of.

None of this means the context is gone for good. It is all still there in the raw session logs, just not where I can reach it mid-conversation. That makes it an indexing problem, the kind with ordinary, well-understood answers. Recall is the safety net whether the loss is Claude's or mine.

<figure class="chart">
					<div class="chart-title">The problem, in one picture</div>
					<img class="chart-img" src="/images/recall-compaction-problem.png" alt="Top row: a session where pre-compaction detail is dropped and replaced by a summary. Bottom row: recall keeps every exchange." />
					<figcaption>Compaction keeps a summary and drops the specifics you were mid-thought on. Recall&rsquo;s local index keeps every exchange verbatim, on both sides of the boundary.</figcaption>
				</figure>

---

## How it works, from the system up

Recall has three parts, all local, with no network involved.

### Capture: a hook on every prompt

Claude Code fires a `UserPromptSubmit` hook before each turn. Recall's hook reads the session transcript, but only the bytes it hasn't seen yet. It keeps a **byte offset per session** and seeks straight to it, so each run costs `O(new turn)` rather than `O(whole transcript)`. That is cheap enough to run on every prompt. It parses the new JSONL into user/assistant *exchanges* and writes them in one transaction. I hit one caveat the hard way. On multi-hundred-megabyte transcripts, an oversized read ran past the hook's 10-second timeout, committed nothing, and re-read the same chunk every time. v2.2.3 caps each read so it banks progress and catches up over a few prompts instead of stalling.

<figure class="chart">
					<div class="chart-title">Capture: how an exchange gets indexed</div>
					<img class="chart-img" src="/images/recall-capture-diagram.png" alt="Flow from every prompt through a hook and an incremental read into a SQLite + FTS5 index holding 2,550 exchanges in a 3.9 MB file." />
					<figcaption>A <code>UserPromptSubmit</code> hook reads only the new bytes since the last turn, parses them into exchanges, and writes them to a local SQLite + FTS5 database. No network, no model calls, no tokens.</figcaption>
				</figure>

### Storage: SQLite + FTS5, verbatim

Everything lands in a single local SQLite database in WAL mode. Exchanges are stored **verbatim** (the exact words, not a summary) behind an **FTS5** full-text index for sub-millisecond search. Every row carries a `session_id` and a hash of the project directory, which is what allows three query ranges: *this session*, *this project* (`--all`), or *everything* (`--global`). A TF-based auto-tagger adds keyword tags as it goes, and a `PreCompact` hook drops in a recovery nudge just before the window compacts.

### Retrieval: a thin command surface

`/recall:recall search <term>` runs FTS5 across the chosen scope. `/recall:recall around <time>` pulls the exchanges from a specific point in the past, so "what were we doing last Tuesday afternoon" resolves to the actual turns from then, and `/recall:recall last5` grabs the most recent. Because every exchange is stored verbatim with its timestamp and project, a query can pin down *which* thing and *when* at the same time, not just a bare keyword. Session identity comes from the native `CLAUDE_CODE_SESSION_ID` that Claude Code injects into every command subprocess, so even with several sessions open on one machine, none of them reads another's history. Capturing costs no cloud, no API calls, and no tokens. The index is a file on my disk.

<figure class="chart">
					<div class="chart-title">Three ways to scope a search</div>
					<img class="chart-img" src="/images/recall-scopes.png" alt="Nested scopes: this session inside this project inside everything." />
					<figcaption>The same query can be pinned to the current session, to every session in this project (<code>--all</code>), or to everything on the machine (<code>--global</code>).</figcaption>
				</figure>

---

## What it costs, and how it scales

A fair question for anything that fires this often is what it costs. On the capture side, close to nothing. The hook reads only the new bytes since the last turn, parses them, and writes to a local file: no network, no model calls, no tokens. On a normal turn that is a few milliseconds and a few kilobytes on disk.

The cost only shows up when I recall, and on a subscription it is not a dollar figure at all. A recall just draws on your usage allowance like any other turn, and its share is small. Measured across my own sessions, one runs to about a thousand tokens of generated output and a couple thousand of injected context; the rest of what a turn weighs, often 150,000 to 200,000 tokens, is the session you already have, re-sent from cache whether or not you recall. Against that, and against re-explaining a thread by hand, a recall is a sliver. Across roughly 470,000 requests spanning all three of my machines, recall is about one percent of the tokens I generate, and under half a percent of my total token flow once the cached context every turn re-sends is counted. On my Max 20x plan, whatever the exact ceiling, it barely moves the needle. It is the deliberate trade, and the reason I keep the plugin.

<figure class="chart">
					<div class="chart-title">What a recall costs a turn</div>
					<img class="chart-img" src="/images/recall-retrieval-cost.png" alt="A retrieval flow feeding a token bar where recall is a ~4k sliver on a ~193k turn." />
					<figcaption>A query runs FTS5 over the chosen scope and injects the matched exchanges into the live turn. That is about 4,000 tokens; the rest is your existing session, re-sent from cache whether or not you recall.</figcaption>
				</figure>

Scale has held up better than I expected. The index keeps only the verbatim exchange text, so it stays small. One of my heaviest machines has folded a 144 MB transcript into a 3.6 MB database of about 2,500 exchanges, and FTS5 search over it still returns in under a millisecond. SQLite is comfortable with millions of rows, so years of use would still measure in the tens of megabytes. The one place that needs care is the first pass over a very large existing transcript, which the hook processes in bounded chunks so it never blocks a prompt.

---

## What I actually reach for it for

Three patterns account for most of my logs.

### The exact command, and when I ran it

Most of my recalls are small but exact. The question is rarely "how do I spin up a RunPod pod," which any model will answer generically. It is "how did *we* spin up *this* pod last Tuesday afternoon", and what comes back is the specific command from the real session, at the time it happened, not a plausible reconstruction. From the log: *"how I spun up the GPU pod, the exact command,"* *"runpod one-liner to spin up / SSH into a pod,"* `search ssh.runpod.io`, `search ssh-add`. Over time it became a **personal ops memory**, and because the store is timestamped and scoped, I can pin a search to *when* as easily as to *what*. That being said, I have had incidents where I have lost entire session histories and have only been able to recover work as a result of the Recall index. Frankly the benefits of the plugin have exceeded the original scope at this point.

### Decisions and terms

Next come decisions and names. Why was something done, and what was that term I coined and then buried under a week of other work? *"What prompted the 'dynamic shapes' roadmap item?"* *"Metal lift for the tridec megakernel, the BLOCK=32 pin."* When you move between five problem spaces, remembering which project decided what, and why, is real work, and recall does it for me.

### Catching Claude up, not just myself

The most useful part is often not recovering context for me, but handing it to Claude. Fully re-explaining where a thread stood can take an hour of typing and back and forth, depending on how lost Claude is. Instead I can say something broad like *"you are forgetting all of the work we covered last week, please catch up on all session context from Monday to Friday"*, or when I need the needle and not the haystack, I get specific: *"catch up on the d=7 decoder benchmarks on the H200 from last Tuesday to Thursday; the launch command for the pod is in there."* A few seconds of retrieval replaces a wall of re-explanation, and Claude picks up sharper than any summary I would have written by hand. It is the same safety net after a compaction or a lost session. When the thread is gone beyond what the summary kept, the verbatim record still has it. That has saved me from redoing hours, sometimes days, of work and agentic research.

<figure class="chart">
					<div class="chart-title">Catching Claude up on a span of time</div>
					<img class="chart-img" src="/images/recall-catch-up.png" alt="A week timeline with a highlighted Tuesday-to-Thursday window fed into a primed Claude." />
					<figcaption>A temporal query pulls the actual turns from a window, last Tuesday to Thursday, straight into the current context, instead of an hour of re-explaining.</figcaption>
				</figure>

---

## Who this is for

If your Claude sessions are lasting more than a week per lifetime, this is for you. If you have ever used an agentic transcription tool, you know the benefit outright. It records a meeting, then answers *"what did we decide about pricing?"* later. Recall is the same idea aimed at your Claude sessions. It passively indexes every exchange as you work, then extracts exactly the piece you ask for. It comes down to volume. A heavy Claude user moves through thousands of exchanges across dozens of sessions, and no one holds every detail of that, however good their memory. If anything, the stronger your memory, the more efficient the plugin is, because you know precisely what to ask for, so it takes fewer tokens to get it. Anyone spending real token budgets across many sessions will want it, whatever their personal recall is like.

---

## How it got here

It didn't start ambitious. **v1**, back in January, was a notepad that held a single session's exchanges in a JSON file, just enough to survive a `/clear`. The one idea that has outlived everything else was already there. Even v1 captured incrementally, by byte offset, keeping each turn cheap. Even that first version had keyword search, date-grouped multi-day sessions, and quick commands like `last5` and `around`, with 91 tests behind them. A point release two weeks later added Claude Cowork support and a proper marketplace install.

The real change came with **v2.0** in April. It rebuilt everything on SQLite and FTS5, with a non-destructive migration off the old JSON. That is what turned "recover my last few messages" into full-text search across *every* session and *every* project, the cross-project reach that makes it useful when the answer lives in a different repo. It also brought TF-based auto-tagging, scope flags for this session, this project, or everything, and session management like `stats`, `prune`, and `export`. **v2.1** added sharing between parallel sessions, highlights and connections on a decaying check schedule; that layer got triggered by accident more than on purpose at first, and cleaning up the mess is what taught me where its edges were. **v2.2** shipped the skill, the first attempt at getting Claude to reach for recall on its own, alongside a performance pass that collapsed a dozen disk syncs per prompt down to one, and a written privacy policy. Across those releases the test suite grew from 91 to nearly 400.

Then came the least glamorous and most necessary stretch. From late June into July, **v2.2.1 through v2.2.3** were spent keeping the plugin alive against a platform that moves. Claude Code had quietly changed the fields its hooks receive, which broke capture with no error at all; the compaction nudge had been registered under an event that no longer fired; and on multi-session Linux boxes a concurrency bug could hand back *another* session's history. Fixing those meant pinning capture to the native per-session id Claude Code now injects. And while measuring my own usage to write this post, I found the capture hook could wedge on multi-hundred-megabyte transcripts and silently skip lines, and that the old event log had been recording test noise instead of real invocations. v2.2.3 fixed the wedge and replaced that log with an invocation counter, which turned the numbers above into a single query instead of a forensic dig. Unglamorous work, but it is the difference between a demo and something you can lean on across five months and three machines.

<figure class="chart">
					<div class="chart-title">The wedge, and the fix</div>
					<img class="chart-img" src="/images/recall-wedge.png" alt="Before: a 10 MB read stuck re-reading. After: 2 MB chunks that bank progress and converge." />
					<figcaption>A single oversized read blew past the hook&rsquo;s 10-second timeout and re-read the same chunk forever. v2.2.3 caps each read so it banks progress and catches up over a few prompts.</figcaption>
				</figure>

## The engine is ahead of the instinct

The plugin and the skill are at different stages. The **plugin** (the capture pipeline, the index, the search) is solid and reliable. The **skill**, the part meant to get Claude to reach for recall on its own, is not there yet. In practice I still drive it by hand, and I am usually the one who types "use `/recall:recall` to catch up." It stays **invocation-driven** for now. Getting Claude to reach for it without being asked is one part of the roadmap. The other is matching by meaning, so a search finds the right exchange even when I don't remember the exact words it used, rather than keyword and time alone.

## How it sits alongside Claude's own memory

Claude has its own memory now, and it is useful. There is Claude Memory on claude.ai, and per-repo notes in Claude Code. Recall does something narrower alongside it. It is a **local, verbatim index** you can full-text search. It also works across sessions and projects, a useful bonus rather than the main point. What matters is that it runs entirely on **your** machine and **your** resources. That means a hook indexing each prompt and a SQLite file on your disk, with no cloud involved. It sits underneath whatever memory Claude already keeps for you, as the local backstop.

<figure class="chart">
					<div class="chart-title">Where recall sits</div>
					<img class="chart-img" src="/images/recall-positioning.png" alt="Three stacked layers: your session, Claude memory, and recall as the bottom backstop." />
					<figcaption>Claude&rsquo;s compaction and memory keep the gist as the window fills. Recall is the local, verbatim backstop underneath, holding what slips past.</figcaption>
				</figure>

---

## How often I use it

In practice, one or two times a day, most days of the week. Most of those are quick catch-ups, a nudge to reorient after a compaction or a dropped thread; only now and then is it a deep dig back through the history for one exchange among many similar ones. The bulk of it has been my Second Nature work.

<figure class="chart">
					<div class="chart-title">The index, filling up</div>
					<img class="chart-img" src="/images/recall-index-growth.png" alt="Cumulative exchanges captured over time, rising to about 2,550 by July." />
					<figcaption>Every exchange, indexed locally as I work, accumulating into a single small database over months. Nothing is summarized away, and all of it stays one query from being recovered.</figcaption>
				</figure>

Those numbers are a floor, for a mundane reason. While building the plugin I reset my own store a few times, so chunks of my personal history are simply gone. That is a side effect of developing the thing, not of using it. The store lives in your home directory, separate from the plugin code, and schema changes are additive migrations, so updating the plugin leaves your history intact. Going forward, v2.2.3 records each invocation as it happens, so the next tally is one `/recall:recall usage` query rather than a hand count.

---

## Going Forward

Frankly I see some strong benefit towards opening this plugin up to other agentic coding harnesses and models, as well as exploring cross-device index sharing. Ultimately I would love to see Anthropic implement this functionality natively within the Claude ecosystem, as I do believe people are willing to opt-in and leverage their own resources for a better Claude experience.

---

`/recall:recall` is a Claude Code and Claude Cowork plugin — local SQLite + FTS5, cross-session and cross-project, no cloud, no API cost. Source: [github.com/bledden/claude-recall-plugin](https://github.com/bledden/claude-recall-plugin).
