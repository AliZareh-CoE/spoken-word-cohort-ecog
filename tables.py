"""The two tables of the paper, every cell read from a result file.

Writes results/tables/table_1_cohort_contribution.tex and table_2_deep_models.tex.
"""
from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

import pandas as pd

import common


def pfmt(p):
    q = Decimal("0.001") if p < 0.1 else Decimal("0.01")
    return str(Decimal(repr(float(p))).quantize(q, rounding=ROUND_HALF_UP))


def _b(x):
    v = x * 1e3
    return f"{v:.3f}" if (f"{v:.2f}" in ("-0.00", "0.00") and v != 0) else f"{v:.2f}"


def _e(x):
    v = x * 1e3
    return f"{v:+.3f}" if (f"{v:.2f}" in ("-0.00", "0.00") and v != 0) else f"{v:+.2f}"

OUT = common.OUT / "tables"
R = common.OUT


def row(lab, r, med="subj_median", p="p_two_sided", n=None, note=""):
    n_el = int(r["n_elec"]) if pd.notna(r.get("n_elec")) else None
    el = f"{n_el:,}".replace(",", "{,}") if n_el is not None else ""
    return f"{lab} & {el} & ${_e(r[med])}$ & $[{_b(r.ci_lo)}, {_b(r.ci_hi)}]$ & {int(r.n_subj_pos)}/{int(r.n_subj)} & ${pfmt(r[p])}${note} \\\\"


a34 = pd.read_csv(R / "permutation_test.csv")
est = pd.read_csv(R / "contribution_size.csv").set_index("electrode_rule")
a31 = pd.read_csv(R / "other_shuffles.csv").set_index("label")
a27 = pd.read_csv(R / "model_free_electrodes.csv").set_index("label")
a25 = pd.read_csv(R / "cohort_contribution.csv").set_index("label")
a26 = pd.read_csv(R / "deep_models.csv").set_index("label")
a28 = pd.read_csv(R / "whisper_past_only.csv").set_index("label")

L = ["\\begin{tabularx}{\\linewidth}{>{\\raggedright\\arraybackslash}Xrrrrr}", "\\toprule", "Analysis & Electrodes & Estimate & 95\\% CI or shuffled sets & Patients $>0$ & $p$ \\\\", "\\midrule",
     "\\multicolumn{6}{l}{\\textit{Test against 205 shuffled sets (one-sided permutation $p$)}} \\\\"]
for r, lab in zip(a34.itertuples(), ("Median of patient medians (primary)", "Mean of patient means")):
    L.append(f"{lab} & {int(r.n_elec)} & ${r.T_real * 1e3:+.3f}$ & ${r.perm_mean * 1e3:+.3f} \\pm {r.perm_sd * 1e3:.3f}$; max ${r.perm_max * 1e3:+.3f}$ & & ${pfmt(r.p_one_sided)}$ \\\\")
L += ["\\midrule", "\\multicolumn{6}{l}{\\textit{Size against the mean of 205 shuffled sets (test across patients)}} \\\\"]
for key, lab in (("original rule (real or mean shuffled CVR > 0.05)", "Speech-responsive electrodes"), ("symmetric set (mean shuffled CVR > 0.05)", "Selected by the shuffled models alone")):
    r = est.loc[key]
    L.append(f"{lab} & {int(r.n_elec)} & ${r.subj_median * 1e3:+.2f}$ & $[{_b(r.ci_lo)}, {_b(r.ci_hi)}]$ & {int(r.n_subj_pos)}/{int(r.n_subj)} & ${pfmt(r.p_wilcoxon)}$ \\\\")
L += ["\\midrule", "\\multicolumn{6}{l}{\\textit{Other shuffles, five shuffled sets each (test across patients, Holm across the two)}} \\\\",
      row("Same position and word length", a31.loc["spectrogram token-level position x length shuffle (position_length), responsive"], p="p_holm"),
      row("Same values for each repeat of a word", a31.loc["spectrogram word-consistent shuffle (word_repeat), responsive"], p="p_holm"),
      "\\midrule", "\\multicolumn{6}{l}{\\textit{Other electrodes, matched shuffle, five shuffled sets (test across patients)}} \\\\",
      row("Selected without the model", a27.loc["spectrogram contribution on model-free electrodes"]),
      row("All electrodes", a25.loc["spectrogram all"]), "\\bottomrule", "\\end{tabularx}"]
(OUT / "table_1_cohort_contribution.tex").write_text("\n".join(L) + "\n")

M = ["\\begin{tabularx}{\\linewidth}{>{\\raggedright\\arraybackslash}Xrrrrr}", "\\toprule", "Model & Electrodes & Estimate & 95\\% CI & Patients $>0$ & $p$ (Holm) \\\\", "\\midrule",
     "\\multicolumn{6}{l}{\\textit{Cohort contribution with one AI model added to the spectrogram model (matched shuffle, five shuffled sets)}} \\\\"]
for key, lab in (("gpt2_current responsive", "$+$ GPT-2, current word"), ("gpt2_previous responsive", "$+$ GPT-2, previous word"), ("whisper_untrained responsive", "$+$ Whisper, untrained"), ("whisper responsive", "$+$ Whisper")):
    M.append(row(lab, a26.loc[key], p="p_holm"))
M += ["\\midrule", "\\multicolumn{6}{l}{\\textit{Whisper restricted in time, added to the spectrogram model}} \\\\"]
M.append(row("$+$ Whisper, past audio only", a28.loc["whisper_past responsive"], p="p_holm"))
M.append(row("$+$ Whisper, 500 ms of later audio", a28.loc["whisper_past_500ms responsive"], p="p_holm"))
M += ["\\midrule", "\\multicolumn{6}{l}{\\textit{Differences between models on a common electrode set}} \\\\"]
for key, lab in (("gpt2_current minus whisper (common set)", "GPT-2, current word, minus Whisper"), ("gpt2_previous minus whisper (common set)", "GPT-2, previous word, minus Whisper"), ("whisper_untrained minus whisper (common set)", "Untrained Whisper minus Whisper")):
    M.append(row(lab, a26.loc[key], p="p_holm"))
M.append(row("Past-only Whisper minus Whisper", a28.loc["whisper_past minus whisper (common set)"], p="p_holm"))
M.append(row("500-ms Whisper minus Whisper", a28.loc["whisper_past_500ms minus whisper (common set)"], p="p_holm"))
M += ["\\midrule", "\\multicolumn{6}{l}{\\textit{Planted signal ($\\rho = 0.04$) recovered by the same test ($p$ uncorrected)}} \\\\",
      row("Spectrogram model", a28.loc["spectrogram planted increment rho 0.04"]), row("$+$ Whisper, past audio only", a28.loc["whisper_past planted increment rho 0.04"]),
      f"Share recovered with the past-only Whisper & & {100 * a28.loc['planted ratio whisper_past/spectrogram', 'subj_median']:.0f}\\% & & & \\\\", "\\bottomrule", "\\end{tabularx}"]
(OUT / "table_2_deep_models.tex").write_text("\n".join(M) + "\n")
print((OUT / "table_1_cohort_contribution.tex").read_text()); print((OUT / "table_2_deep_models.tex").read_text())
