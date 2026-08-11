# Full-scale run on RunPod

Two-machine split, not one long-running pod:

| Phase | Where | Why |
|---|---|---|
| 0-2 (env, load 100 cases, PRIMERA generation) | **RunPod RTX 4090**, one session, a few hours | Only GPU-bound step. Pay only while it runs. |
| 3-5 (extract, verify, correct, evaluate) | **Local M1**, free, spread over ~3-5 days | CourtListener caps at 125 requests/day — a GPU pod sitting idle across that wait would just burn money for nothing. |

Config: [`config/experiment_test100.yaml`](../config/experiment_test100.yaml) — 100 citation-rich cases (3-36 gold citations each, 631 total) from the **test** split, D→L only. Selected via `scripts/scan_citation_rich_cases.py`.

Driven manually from the pod's Jupyter UI (not via RunPod's API — no `RUNPOD_API_KEY` needed).

## 1. Get the code onto the pod

Locally:
```bash
cd /Users/rajakarthikchirumamilla/Desktop/Thesis
zip -r code_bundle.zip code \
  -x "code/.venv/*" -x "code/data/hf/*" -x "code/data/multilexsum/*" \
  -x "code/outputs/*" -x "*/__pycache__/*" -x "code/.env" \
  -x "code/*.egg-info/*" -x "code/colab_upload/*"
```
Upload `code_bundle.zip` into the Jupyter file browser, into the **same folder** as where you'll open the notebook below (drag-and-drop, or the upload button). Don't unzip it yet — the notebook's first cell does that.

## 2. Run generation — notebook

Open `notebooks/runpod_generation.ipynb` (already inside the zip, at `code/notebooks/runpod_generation.ipynb` — move or copy it next to `code_bundle.zip` first, or just open it from there directly) and run cells top to bottom:

| Cell | Does |
|---|---|
| 1 | Unzips `code_bundle.zip`, sets up `sys.path` |
| 2 | Confirms the GPU is visible (`nvidia-smi` + `torch.cuda.is_available()`) |
| 3 | `pip install -r requirements-colab.txt` (numpy pinned `<2.0` — see the file header for why) |
| 4 | Prompts for `HF_TOKEN` via a masked `getpass()` input — never written into the notebook file. Only `HF_TOKEN` is needed on the pod; `COURTLISTENER_API_TOKEN` stays on your laptop since Module C runs there |
| 5 | Pre-flight smoke checks (env, secrets, dataset load, citation extraction) — fails fast with a clear message if something's wrong, before any GPU time is spent |
| 6 | Loads the 100 selected test-split cases |
| 7 | **Generates** — the long-running cell, checkpoints to `outputs/generations_test100/test_long.jsonl` after every case. If it dies partway (kernel restart, pod hiccup), just re-run the cell — already-written `case_id`s are skipped, nothing repeats |
| 8 | Prints what to download and reminds you to terminate the pod |

Regenerate this notebook anytime with `python scripts/build_runpod_generation_notebook.py` if the underlying scripts change.

Expect roughly 1-3 hours for 100 D→L cases on an RTX 4090 (beam=5, ~850 output tokens/case).

*(A terminal-equivalent also exists — `scripts/run_generation_runpod.py`, same logic, run via `tmux` — in case a cell ever needs debugging outside the notebook.)*

## 3. Bring the results home, then kill the pod

In the Jupyter file browser, navigate to `code/outputs/generations_test100/test_long.jsonl` and download it directly (small file — a few MB for 100 summaries). Save it locally to `code/outputs/generations_test100/test_long.jsonl` (create the folder if needed).

**Then stop/terminate the pod** — that's the only step costing money, and generation is the only phase that needs the GPU.

## 4. Local verification (multi-day, free)

```bash
cd code && source .venv/bin/activate
python scripts/run_verification_local.py --config experiment_test100.yaml
```
Resume-safe: stops cleanly when CourtListener's daily cap is hit (tracked in `data/processed/rate_limiter_state.json`, persists across runs), regenerates `outputs/results_test100/rq_answers.md` and friends on whatever's done so far each time. Re-run it once a day until it reports "All 100 cases verified."
