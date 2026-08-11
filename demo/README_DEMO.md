# Professor Pipeline Demo App (Canvas + Live Showcase)

## What you are seeing

1. **Cursor Canvas**: a polished **replay** of the full dual-system pipeline on **10 citation-rich cases** using the saved artifacts from `full_pipeline_results/`.
2. **Streamlit live tab**: a **one-case** on-demand run for **SD-KS-0001**, using real eyecite extraction + CourtListener verification + symbolic correction.

The Canvas makes **no network calls** during the demo (everything is embedded at build time).

## Build / regenerate the Canvas

From the `code/` folder:

```bash
cd /Users/rajakarthikchirumamilla/Desktop/Thesis/code

python scripts/bundle_demo_data.py \
  --results-dir /Users/rajakarthikchirumamilla/Desktop/Thesis/full_pipeline_results \
  --output demo/demo_bundle.json \
  --draft-chars 2400 \
  --gold-chars 450

python scripts/build_demo_canvas.py \
  --bundle demo/demo_bundle.json
```

Then open the managed canvas file in Cursor:

`thesis-pipeline-demo.canvas.tsx`
in:
`~/.cursor/projects/Users-rajakarthikchirumamilla-Desktop-Thesis/canvases/`

## Live showcase (Streamlit)

1. Install Streamlit (demo only):

```bash
pip install streamlit
```

2. Export the CourtListener token and run:

```bash
export COURTLISTENER_API_TOKEN="YOUR_TOKEN"
cd /Users/rajakarthikchirumamilla/Desktop/Thesis/code
streamlit run demo/live_showcase.py
```

The live app is for **SD-KS-0001** and runs:
`Extract citations -> Verify (CourtListener) -> Symbolic correct`.

## Suggested cases to click (today)

- **Success walkthrough**: `SD-KS-0001` (Brown v. Board) — symbolic correction drives HR from ~1.0 to ~0.0 for extracted cites.
- **Failure walkthrough**: `PB-CA-0016` — HR stays ~1.0 (symbolic correction cannot fully resolve the cited failures).

## 60-second professor script

Opening (30s):  
“We built a dual-system pipeline for legal summarization reliability: generate a draft summary, extract citations, verify them against CourtListener, then apply a symbolic correction pass. Here are 10 citation-rich validation cases.”

Click SD-KS-0001 (20s):  
“Start with Brown v. Board. The draft contains extracted citations that fail verification. After symbolic correction, the final summary removes or replaces unverifiable reporter strings, improving the hallucination reliability for citations.”

Dashboard takeaway (10s):  
“Across these 10 cases, the baseline citation hallucination rate is about 85.7%, and symbolic correction reduces it to about 66.7%, a relative reduction of about 33%. We also track summary quality with ROUGE-L to ensure correction does not collapse readability.”

Honest limits (brief):  
“This is preliminary n=10 on the validation split, and CourtListener calls are rate-limited; the Canvas itself is a replay, while the Streamlit tab shows a real on-demand run.”

