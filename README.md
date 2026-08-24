# Draft War Room

A free, no-server live draft assistant for fantasy football. Upload your own
rankings CSV and follow along with your snake draft in real time:

- **Best available, sorted by YOUR rankings** — not the platform's
- **Live pick ticker** — how many picks until you're up, with on-the-clock urgency
- **Value alerts** — flags players falling past their ADP and warns when the room overpays
- **Positional run detection** — know when a QB/RB/WR/TE run just started
- **Roster deadline rules** — construction guardrails derived from a 5-season
  study of championship rosters in a competitive 16-team full-PPR league
  (togglable for other formats)
- **Auto-save** — everything persists in your browser via localStorage; close
  the tab mid-draft and pick up where you left off

No accounts. No backend. Your rankings never leave your browser.

## Quick start (local)

```bash
npm install
npm run dev
```

Open the printed localhost URL, upload a rankings CSV, set your league size and
draft slot in Setup, and you're live.

## CSV format

A header row with these columns (any order; extra columns are ignored):

```
Player, Position, Team, Rank, ADP
```

- `Rank` can also be named `ETR Rank`, `My Rank`, or `Overall Rank`
- Positions: QB, RB, WR, TE, DST (or D/ST, DEF), K (or PK)
- `Team` and `ADP` are optional — without ADP you lose the value badges, but
  everything else works
- A sample template lives at `public/sample-rankings.csv`

## During the draft

| Action | What it does |
|---|---|
| **GONE** | Logs someone else's pick, advances the clock one pick |
| **MINE** | Logs your pick and **auto-syncs the clock to your scheduled slot** — even if you missed logging a few picks, drafting realigns everything |
| **SKIP** | Advances the clock without logging a player |
| **SYNC** | Type the pick number your draft platform shows; the clock jumps there |
| **UNDO** | Reverses the last action, clock included |

## Deploy

**Vercel / Netlify:** import the repo, framework preset "Vite", build command
`npm run build`, output directory `dist`. Done.

**GitHub Pages:** set `base: '/<your-repo-name>/'` in `vite.config.js`, then:

```bash
npm run build
# publish the dist/ folder — e.g. with the gh-pages package:
npx gh-pages -d dist
```

## Notes

- The bundled deadline rules reflect one league's championship history
  (16-team full PPR, 1QB). They generalize reasonably to 10–14 team PPR
  leagues but should be toggled off for superflex/2QB formats.
- Rankings data is yours: bring any source you're licensed to use. This
  repo intentionally ships with no rankings data.

## License

MIT — use it, fork it, bring it to your league.
