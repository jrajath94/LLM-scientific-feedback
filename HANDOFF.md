# AI Reviewer Handoff — Finish Building This

## Goal
A globally-installable CLI tool (`ai-review`) that produces **NeurIPS-grade peer reviews** for any PDF, using the MiniMax API, with a **swarm council** of reviewers (theorist + empiricist + skeptic + area chair). Usable across all projects on this machine.

---

## Current State (40% done)

### Already Done
- **Forked repo:** `Weixin-Liang/LLM-scientific-feedback` → `jrajath94/LLM-scientific-feedback`
- **Cloned to:** `~/ai-reviewer/`
- **Patched `~/ai-reviewer/main.py`:** `GPT4Wrapper` class now calls MiniMax API (`https://api.minimaxi.com/v1/text/chatcompletion_v2`) instead of OpenAI. Reads key from `MINIMAX_API_KEY` env var OR `~/.minimax_key` file.
- **Working reference script** at `/Users/rj/research-claw/papers/semcp-conformal-prediction/ai_reviewer.py` (already has a working 3-reviewer + meta-review swarm using MiniMax — use this as the swarm template).

### MiniMax API Details
- **Endpoint:** `https://api.minimaxi.com/v1/text/chatcompletion_v2`
- **Model:** `MiniMax-M2` (or `MiniMax-M2.5`)
- **Auth:** `Authorization: Bearer <key>`
- **Payload:** `{"model", "messages", "max_tokens", "temperature"}`
- **Key to use:**
  ```
  sk-cp-o1xzlbG-DipkSq4KhiIVaAOADMUsZMoqzuQe6sbOc2xa4kKljWpFPt3F8oMY15KtIn9B3kiaJjic8pWCRjfpOEdBlMzJTYfPcdSDdG_rvvCXV8AHiK5NjCA
  ```
  **NOTE:** When first tested earlier, the API returned `{"status_code":2049,"status_msg":"invalid api key"}`. User confirmed this is **MiniMax International**. The key format `sk-cp-...` is unusual for MiniMax — first task is to verify which endpoint/auth format this key actually needs. Try:
  - `https://api.minimaxi.com/v1/text/chatcompletion_v2` (v2 International)
  - `https://api.minimax.chat/v1/text/chatcompletion_pro` (China)
  - Check MiniMax International console: https://www.minimax.io/platform_overview
  - If still 2049 error, ask user to re-check or try with `X-Api-Key` header instead of `Bearer`.

---

## Remaining Work (60%)

### Step 1: Fix `main.py` residual imports (5 min)
File: `~/ai-reviewer/main.py`

- Remove remaining references to `openai`, `tiktoken`, `gradio` (grep the file).
- Make sure `wrapper = GPT4Wrapper(...)` at module bottom only runs if key is set (lazy init, wrap in try/except or move inside a function).
- The `step3_get_lm_review(parsed_xml)` function should still work — it uses `wrapper.send_query()`.

### Step 2: Save key securely (1 min)
```bash
echo -n "sk-cp-o1xzlbG-DipkSq4KhiIVaAOADMUsZMoqzuQe6sbOc2xa4kKljWpFPt3F8oMY15KtIn9B3kiaJjic8pWCRjfpOEdBlMzJTYfPcdSDdG_rvvCXV8AHiK5NjCA" > ~/.minimax_key
chmod 600 ~/.minimax_key
```

### Step 3: Build the swarm council CLI (20 min)
Create `~/ai-reviewer/ai_review_cli.py`:

```python
"""ai-review: NeurIPS-grade swarm-council reviewer CLI.

Usage: ai-review paper.pdf [--model MiniMax-M2] [--out reviews.md]

Runs 4 reviewers (theorist, empiricist, skeptic, originality-assessor)
then an Area Chair meta-review. Each reviewer follows the exact NeurIPS
2025 template with scores on Quality, Clarity, Significance, Originality,
Soundness, Presentation, Contribution, Confidence, Overall (1-6).
"""
import argparse, json, os, sys
from pathlib import Path
import fitz  # pymupdf
import requests

API = "https://api.minimaxi.com/v1/text/chatcompletion_v2"

# Copy the NEURIPS_REVIEWER_PROMPT verbatim from:
#   /Users/rj/research-claw/papers/semcp-conformal-prediction/ai_reviewer.py
# It's already NeurIPS-grade with the 9-score template.

PERSONAS = {
    "theorist": "You emphasize theoretical rigor: theorem statements, proof validity, assumptions, mathematical precision. You penalize unjustified claims and reward elegant proofs.",
    "empiricist": "You emphasize experimental rigor: benchmark coverage, baseline strength, statistical significance, reproducibility. You penalize weak evaluations and cherry-picking.",
    "skeptic": "You are an adversarial domain expert looking for overclaims, missing comparisons, weak scholarship, and fabricated numbers. You reward honest negative results.",
    "originality": "You assess novelty vs prior work. You know the field deeply and identify what is genuinely new vs incremental repackaging.",
}

AREA_CHAIR_PROMPT = """You are the NeurIPS 2026 Area Chair. Four expert
reviewers submitted reviews. Write a decisive meta-review that:
1. Summarizes consensus + divergences
2. Identifies the 3 most important issues
3. Gives final accept/reject with score X/6
4. Lists conditions under which paper would be accepted
Area chairs do not hedge. Be decisive."""


def load_key():
    key = os.environ.get("MINIMAX_API_KEY", "").strip()
    if not key:
        path = os.path.expanduser("~/.minimax_key")
        if os.path.exists(path):
            key = open(path).read().strip()
    if not key:
        sys.exit("ERROR: set MINIMAX_API_KEY or create ~/.minimax_key")
    return key


def extract_pdf(pdf_path, max_pages=20):
    doc = fitz.open(pdf_path)
    return "\n\n".join(
        f"=== Page {i+1} ===\n{p.get_text()}" for i, p in enumerate(doc) if i < max_pages
    )


def call_minimax(messages, key, model, max_tokens=8192):
    resp = requests.post(
        API,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": model, "messages": messages,
              "max_tokens": max_tokens, "temperature": 0.3},
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--model", default="MiniMax-M2")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    key = load_key()
    paper = extract_pdf(args.pdf)
    print(f"Extracted {len(paper):,} chars from {args.pdf}")

    # NEURIPS_REVIEWER_PROMPT - copy from ai_reviewer.py
    from ai_reviewer_prompt import NEURIPS_REVIEWER_PROMPT  # put prompt in a separate file

    reviews = {}
    for name, persona in PERSONAS.items():
        print(f"\n[{name.upper()}] reviewing...")
        sys_prompt = NEURIPS_REVIEWER_PROMPT + f"\n\n**Persona**: {persona}"
        msgs = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Review this paper:\n\n{paper}"},
        ]
        reviews[name] = call_minimax(msgs, key, args.model)
        print(f"  done ({len(reviews[name]):,} chars)")

    # Meta-review
    print("\n[AREA CHAIR] synthesizing...")
    joined = "\n\n---\n\n".join(f"## Reviewer {n}\n\n{r}" for n, r in reviews.items())
    meta = call_minimax(
        [{"role": "system", "content": AREA_CHAIR_PROMPT},
         {"role": "user", "content": joined}],
        key, args.model,
    )

    # Write output
    out = args.out or str(Path(args.pdf).with_suffix(".reviews.md"))
    with open(out, "w") as f:
        f.write(f"# NeurIPS Swarm Review — {args.pdf}\n\n")
        for n, r in reviews.items():
            f.write(f"\n---\n\n## Reviewer: {n.capitalize()}\n\n{r}\n")
        f.write(f"\n---\n\n## Area Chair Meta-Review\n\n{meta}\n")
    print(f"\nSaved: {out}")
    print("\n" + "=" * 60)
    print("META-REVIEW")
    print("=" * 60)
    print(meta)


if __name__ == "__main__":
    main()
```

### Step 4: Copy the NeurIPS prompt file (1 min)
```bash
cp /Users/rj/research-claw/papers/semcp-conformal-prediction/ai_reviewer.py \
   ~/ai-reviewer/ai_reviewer_prompt.py
# Then edit that file to only keep the NEURIPS_REVIEWER_PROMPT constant
```

Or simpler — just paste the `NEURIPS_REVIEWER_PROMPT` string directly into `ai_review_cli.py`. It's ~100 lines, worth the inline.

### Step 5: Make it installable globally (5 min)
Create `~/ai-reviewer/pyproject.toml`:

```toml
[project]
name = "ai-reviewer"
version = "0.1.0"
description = "NeurIPS-grade swarm-council paper reviewer using MiniMax"
requires-python = ">=3.9"
dependencies = [
    "pymupdf>=1.23",
    "requests>=2.31",
]

[project.scripts]
ai-review = "ai_review_cli:main"

[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"
```

Install globally:
```bash
cd ~/ai-reviewer
/Users/rj/opt/anaconda3/bin/pip install -e .
# Verify:
which ai-review
ai-review --help
```

### Step 6: Test on the SemCP paper (5 min)
```bash
ai-review /Users/rj/research-claw/papers/semcp-conformal-prediction/artifacts/deliverables/paper.pdf
```

Expected output:
- Console: progress for 4 reviewers + area chair
- File: `paper.reviews.md` next to the PDF
- Total time: ~3-5 min for all 5 calls
- Total cost: ~60K-100K tokens (~$0.05-0.10 on MiniMax)

### Step 7: Commit + push (2 min)
```bash
cd ~/ai-reviewer
git add -A
git commit -m "feat: MiniMax-backed swarm council reviewer CLI

- Replaced OpenAI wrapper with MiniMax API
- Added 4-reviewer swarm (theorist/empiricist/skeptic/originality) + area chair
- Installable CLI: pip install -e . → ai-review
- Produces NeurIPS-template reviews with 9-score rubric"
git push origin main
```

---

## Verification Checklist

- [ ] `~/.minimax_key` exists with 600 permissions
- [ ] `which ai-review` returns a path
- [ ] `ai-review --help` prints usage
- [ ] `ai-review <some-paper.pdf>` runs without errors
- [ ] Output file contains 4 full reviews + meta-review
- [ ] Each review has all 9 scores filled in
- [ ] Reviews cite specific line numbers from the paper
- [ ] Meta-review gives a decisive X/6 recommendation

---

## Key Files

| File | Purpose |
|------|---------|
| `~/ai-reviewer/main.py` | Stanford's original + MiniMax patch (40% done) |
| `~/ai-reviewer/ai_review_cli.py` | **NEW** — swarm council CLI (to be built) |
| `~/ai-reviewer/ai_reviewer_prompt.py` | **NEW** — NeurIPS prompt constant |
| `~/ai-reviewer/pyproject.toml` | **NEW** — install config |
| `~/.minimax_key` | API key (chmod 600) |
| `/Users/rj/research-claw/papers/semcp-conformal-prediction/ai_reviewer.py` | Working reference swarm (copy prompts from here) |

---

## If the MiniMax Key Fails

1. First verify with curl:
   ```bash
   curl -X POST https://api.minimaxi.com/v1/text/chatcompletion_v2 \
     -H "Authorization: Bearer $(cat ~/.minimax_key)" \
     -H "Content-Type: application/json" \
     -d '{"model":"MiniMax-M2","messages":[{"role":"user","content":"hi"}],"max_tokens":10}'
   ```
2. If `{"status_code":2049,"status_msg":"invalid api key"}`, the `sk-cp-` prefix suggests this might be a different provider's key (possibly OpenRouter, Anthropic proxy, or a key from `enjoyclaudecode` / `apeX` proxy). Check with user.
3. Alternative endpoints to try:
   - `https://api.minimax.chat/v1/text/chatcompletion_pro` (China)
   - `https://api.minimaxi.com/v1/text/chatcompletion_pro` (International, v1 legacy)
4. Alternative auth header: `X-API-Key: <key>` instead of `Authorization: Bearer`
5. If key confirmed wrong, ask user for correct MiniMax International key (should start with `eyJ...` for JWT or similar).

---

## Once Working — Additional Enhancements (Optional)

1. **Cache PDF extraction** — hash the PDF, skip re-extraction on re-runs
2. **Add OpenReviewer** as a second backend option (fork `maxidl/openreviewer`, run its fine-tuned model locally for comparison)
3. **Cross-project memory** — log reviews to `~/.ai-reviewer/history.jsonl` so you can track patterns across papers
4. **MCP server wrapper** — wrap the CLI as an MCP server so Claude Code can invoke it via `ai-review` tool in any project

For the MCP wrapper, create `~/ai-reviewer/mcp_server.py`:
- Use `mcp.server.Server` from the MCP Python SDK
- Expose one tool: `review_paper(pdf_path: str) -> review_markdown`
- Register in `~/.claude.json` under `mcpServers`

---

## Success Criterion

After setup, from ANY project directory you should be able to run:
```bash
ai-review /any/path/to/paper.pdf
```
And get a NeurIPS-quality review within 5 minutes, saved to `paper.reviews.md`.

Total setup time from this handoff: **~40 minutes**.
