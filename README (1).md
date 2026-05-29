# The Uncertainty of Thought — Plot reproduction

Code to regenerate the 16 figures from
**"The Uncertainty of Thought: Distinguishing the Impact of Surprisal vs. Entropy on Human Reading Times"**
(Chandak & Gupta, IIIT Hyderabad, 2025).

Three scripts, one block of figures each:

| Script | Figures | What it does |
|---|---|---|
| `h1_h2_natural_stories.py` | 1 – 8 | H1 (entropy adds signal) and H2 (high H slows readers) on Natural Stories. Includes marginal regressions, LRT, β coefficients, BIC ladder, S × H quadrants, GMM, conditional H. |
| `h3_garden_path.py` | 9 – 13 | H3 garden-path analyses: per-item H/S curves, lead-time distribution, grand-averaged cascade, ΔH at disambiguation, full H → RT → S cascade. Also reports Wilcoxon and cross-lag stats from §5.5. |
| `dundee_replication.py` | 14 – 16 | Cross-corpus replication on Dundee: NS-vs-Dundee β, quadrant replication + Cohen's *d*, first-pass vs total fixation, cross-corpus β scatter. |

## Setup

```bash
pip install -r requirements.txt
```

## Data layout

Default expected paths (override via CLI flags):

```
data/
  natural_stories.parquet     # word-level RT + features
  dundee.parquet              # word-level fixation durations + features
  garden_path.parquet         # per-(item, position) features for 500 GP + 500 controls
```

### Schemas

**`natural_stories.parquet`** — one row per word event
| column | type | notes |
|---|---|---|
| `subject` | str/int | participant id |
| `item` | str/int | story id |
| `word_id` | int | sequential index in item |
| `RT` | float | self-paced RT, ms |
| `surprisal` | float | −log P(w \| ctx) |
| `entropy` | float | H of next-word distribution |
| `delta_H` | float | H(t−1) − H(t) |
| `word_length` | int | characters |
| `log_freq` | float | log unigram freq |
| `position` | int | position in sentence |

**`dundee.parquet`** — same, but with `first_pass_RT` and (optionally) `total_fixation_RT` instead of `RT`.

**`garden_path.parquet`** — long format, one row per (item, position)
| column | type | notes |
|---|---|---|
| `item_id` | str/int | unique sentence id |
| `is_garden_path` | 0/1 | 1 = GP, 0 = matched control |
| `construction` | str | `object_RC`, `subject_RC`, `NPZ`, `main_verb` |
| `position` | int | 0-indexed token position |
| `disambig_pos` | int | position of disambiguating word for this item |
| `word` | str | token (optional, used for x-tick labels in Fig 9) |
| `H` | float | entropy at this position |
| `S` | float | surprisal at this position |
| `delta_H` | float | optional; computed if absent |
| `RT` | float | optional; required only for Fig 13 |

## Running

```bash
# Figures 1–8
python h1_h2_natural_stories.py --data data/natural_stories.parquet --out figures/

# Figures 9–13
python h3_garden_path.py --data data/garden_path.parquet --out figures/

# Figures 14–16
python dundee_replication.py \
    --ns data/natural_stories.parquet \
    --dundee data/dundee.parquet \
    --out figures/
```

Outputs are PNGs at 200 DPI, named `fig01_*.png` … `fig16_*.png`.

### Notes

- `h1_h2_natural_stories.py` defaults to OLS approximations for the LRT (fast).
  Pass `--use-lmm` to refit the by-subject + by-item mixed-effects models that
  produce the χ² values reported in the paper. This is much slower on the full
  Natural Stories corpus.
- The GMM in Fig 7 selects *k* by BIC on the range `[2, 6]`. Component labels
  (Fast/Normal/Slow/Very Slow) are assigned by ascending mean log(RT).
- For Fig 9 you can pin specific items: `python h3_garden_path.py --items NS003 NS017 NS128 NS302`.
