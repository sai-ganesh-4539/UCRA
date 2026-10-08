# UCRA project site — build & deploy guide

A zero-backend, single-page Next.js site showcasing the UCRA project.
All numbers are precomputed from the repo's `outputs/*.json` — no database,
no API keys, no login. Everything renders client-side.

---

## What's inside

```
ucra-site/
├── src/
│   ├── app/page.tsx            # the single page (section order lives here)
│   ├── components/ucra/        # all sections + interactive labs
│   │   ├── hero.tsx            # headline + key stats
│   │   ├── problem.tsx         # why static reservations fail
│   │   ├── results.tsx         # baseline comparison + metric toggle
│   │   ├── kappa-lab.tsx       # interactive k (risk dial) slider lab
│   │   ├── drift-demo.tsx      # "Inject +35% surge" 2-phase animation
│   │   ├── cttc.tsx            # per-slice results w/ slice selector
│   │   ├── pipeline.tsx        # 5-stage architecture walkthrough
│   │   ├── gallery.tsx         # repo figures
│   │   └── repro.tsx           # reproduce-it commands + datasets
│   ├── lib/ucra.ts             # typed data module (imports the JSONs below)
│   └── lib/*.json              # real outputs copied from E:\UCRA\outputs
├── public/figures/             # PNG figures from outputs/
└── next.config.ts              # preconfigured for static export
```

---

## Option A — just look at it (nothing to install)

The site is already live in this workspace. Open the **preview link** in this
chat. Download this folder only if you want to host it yourself.

---

## Option B — run it on your Windows machine (2 minutes)

Prereq: [Node.js 20+ LTS](https://nodejs.org) (comes with npm).

```powershell
cd path\to\ucra-site
npm install
npm run dev          # http://localhost:3000
```

---

## Option C — host it free on Vercel (recommended)

1. Push this folder to a GitHub repo (or use "Add New Project" on vercel.com
   and upload).
2. Go to https://vercel.com → New Project → import the repo.
3. Framework preset auto-detects "Next.js". Click **Deploy**. Done —
   you get a `https://<name>.vercel.app` URL.

No config changes needed. `next.config.ts` with `output: "export"` works on
Vercel as-is.

---

## Option D — GitHub Pages

The build emits a fully static `out/` folder.

1. Install deps and set your repo name in `next.config.ts`:

   ```ts
   basePath: "/UCRA",   // <- your GitHub repo name, e.g. "UCRA"
   ```

   (Only needed for a *project* site `user.github.io/<repo>/`; skip for a
   *user* site `user.github.io`.)

2. Build:

   ```powershell
   npm install
   npm run build       # produces out/
   ```

3. Publish `out/` to the `gh-pages` branch:

   ```powershell
   npx gh-pages -d out -t true
   ```

4. On GitHub: **Settings → Pages → Source: `gh-pages` branch / root**.
   Your site: `https://<user>.github.io/UCRA/`.

Tip: to preview the production build locally first:

```powershell
npm run start        # serves out/ at http://localhost:3000
```

---

## Updating the numbers

All headline metrics live in `src/lib/*.json` (copied verbatim from the
UCRA repo's `outputs/`). After re-running experiments:

1. Copy the new `ucra_results.json`, `cttc_results.json`, `drift_demo.json`,
   `train_metrics.json` into `src/lib/`.
2. Rebuild figures (`python scripts/make_figures.py` in the UCRA repo) and
   copy the PNGs into `public/figures/`.
3. If the per-slot shapes changed, regenerate `ucra-series.json` with the
   repo's `build_site_data.py` helper.
4. `npm run build` again.

---

## Verified

Tested in a real browser before packaging:

- Hero, all 9 sections render; sticky nav anchors work; footer sticks.
- Results metric toggle (Safety / Efficiency / Utilization) re-sorts bars.
- k-lab slider moves the reservation rule R = q0.9 + k·spread and the
  violation/utilization/over-provision stats update live (k = 0 … 2).
- Drift demo: "Inject +35% surge" → frozen model collapses at slot 37;
  "Enable Stage-5 self-evolution" → reservation re-hugs demand (updates at
  slots 96/120/144); Replay resets.
- CTTC slice selector (URLLC / eMBB / mMTC) switches test snapshots
  (n = 577 / 337 / 630) and both metrics.
- Mobile (390px): nav collapses, sections stack, charts scale.
