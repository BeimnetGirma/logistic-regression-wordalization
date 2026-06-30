# Decision Axes: Paper Definition vs Code Implementation

Audit of Axes 1 and 2, how each combination of Scale and Risk Reference is implemented in the codebase and mapped to the paper formulas.

---

## Table 1: Contribution calculation (Scale axis only)

The contribution formula is determined by Scale alone.

| Scale | `raw` defined as | Contribution code |Paper formula |
|---|---|---|---|
| **Log-odds** | `df[param] * β` | `raw - raw.mean()`| `x_j·β_j − mean(x_j·β_j)` |
| **Odds Ratio** | `df[param] * β` | `np.exp(raw - raw.mean())` | `exp(x_j·β_j − mean(x_j·β_j))` |

---

## Table 2: Threshold calculation and LLM text use (Scale × Risk Reference)

**Notation:**
- `k` = most variable feature (for Global strategy)
- `j` = feature being described (for Feature-specific strategy)
- `J` = total number of features
- `σ` = standard deviation computed over all individuals in the dataset
- In odds space, `σ` is computed in log-space then exponentiated back

### Log-odds scale

| Risk Reference | Threshold code | Resulting thresholds | Paper formula | Synthetic  text use |
|---|---|---|---|---|
| **Global** | `mean, std = mean(data_k), std(data_k)` → `[mean + i·std for i in −1, −0.5, 0.5, 1]` | `[−σ_k, −0.5σ_k, 0.5σ_k, σ_k]` (mean ≈ 0 since contributions are mean-centred) | *"Wording thresholds (±0.5σ, ±1σ) defined based on the dominant feature, applied uniformly"* | Compare `contribution_j` to these 4 values → tier word (e.g. *"implies a moderately increased risk"*)  |
| **Feature-specific** | `mean, std = mean(data_j), std(data_j)` → `[mean + i·std for i in −1, −0.5, 0.5, 1]` | `[−σ_j, −0.5σ_j, 0.5σ_j, σ_j]` per feature | *"Thresholds defined separately for each feature using its own contribution distribution"* | Same tier comparison using per-feature bins  |
| **Average** | Mean across all feature-specific bins (excluding `total_risk_contribution`) | `[(1/J)·Σ_j(i·σ_j)]` for each position `i` | *"Evaluates risk relative to the typical effect size across statistically significant features"* | Same tier comparison using averaged bins |

### Odds Ratio scale

| Risk Reference | Threshold code | Resulting thresholds | Paper formula | Synthetic  text use |
|---|---|---|---|---|
| **Global** | `log_data = log(data_k)` → `log_mean, log_std` → `[exp(log_mean + i·log_std) for i in −1, −0.5, 0.5, 1]`  | `[exp(−σ_k), exp(−0.5σ_k), exp(0.5σ_k), exp(σ_k)]` | Same anchor as log-odds global, exponentiated into odds space | **Not used in text.** Percentage computed directly as `(contribution_j − 1) × 100` |
| **Feature-specific** | `log_data = log(data_j)` → `log_mean, log_std` → `[exp(log_mean + i·log_std) for i in −1, −0.5, 0.5, 1]` |  `[exp(−σ_j), exp(−0.5σ_j), exp(0.5σ_j), exp(σ_j)]` per feature | Same paper definition, exponentiated | **Not used in  text.** Same direct percentage as prev |
| **Average** | Mean across all feature-specific odds bins (excluding `total_risk_contribution`) | `[(1/J)·Σ_j exp(i·σ_j)]` for each position `i` | Same paper definition, exponentiated | **Not used in text.** Same direct percentage as previous |

---

## Important Bit (as per our last conversation)

The log-odds and odds thresholds are the same underlying σ values, just expressed in different spaces:

| Threshold position | Log-odds | Odds Ratio |
|---|---|---|
| Lower strong | `−σ_k` | `exp(−σ_k)` |
| Lower moderate | `−0.5σ_k` | `exp(−0.5σ_k)` |
|  |etc |  |



