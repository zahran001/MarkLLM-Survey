# Deploying the Landing Page to GitHub Pages

## Setup (one time)

1. **Figure images are already in `assets/`:**
   - `assets/detectability-1.png` — detectability across model scales (TPR at 10% FPR)
   - `assets/paraphrase-1.png` — surviving detection after GPT-3.5 paraphrasing
   - `assets/robustness-1.png` and `assets/quality_ppl-1.png` — additional charts (not currently embedded in the landing page but available for future additions)

2. **Commit and push:**
   ```bash
   git add index.html assets/
   git commit -m "Add GitHub Pages landing site"
   git push
   ```

3. **Enable GitHub Pages:**
   - Go to the repo on GitHub → **Settings** → **Pages**
   - Under *Source*, select **Deploy from a branch**
   - Branch: **main** / Folder: **/ (root)**
   - Click **Save**

4. **Wait ~1 minute**, then visit:
   ```
   https://zahran001.github.io/MarkLLM-Survey/
   ```

## After deploying

- Update the repo's **About** sidebar (top-right on the repo homepage, click the gear icon) and paste the Pages URL into the **Website** field. This makes the site discoverable directly from the repo.

## Updating the page

Any `git push` to `main` automatically re-deploys within ~1 minute. No manual step needed after the first setup.
