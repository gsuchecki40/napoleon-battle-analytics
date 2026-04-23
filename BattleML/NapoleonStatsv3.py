import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats

# ── Load & prep ───────────────────────────────────────────────────────────────
bel = pd.read_csv('./BattleML/CDB90/data/belligerents.csv')
df  = pd.read_csv('./BattleML/data/battles_clustered.csv')

bel['co_clean'] = bel['co'].replace({
    'BONAPARTE':             'NAPOLEON I',
    'WELLINGTON & BLUECHER': 'WELLINGTON',
})

bel_merged = bel.merge(df[['isqno', 'kmeans', 'casualty_intensity',
                             'force_ratio', 'attacker_underdog']], on='isqno', how='left')

cluster_names = {
    0: "Large-Scale Attritional",
    1: "High-Intensity Defensive",
    2: "Decisive Pursuit",
    3: "Small-Scale Engagement",
    4: "High-Intensity Offensive",
    5: "Massive Set-Piece",
    6: "Failed Assault",
    7: "Operational-Scale Annihilation",
}

generals = ['NAPOLEON I', 'FREDERICK II', 'LEE', 'WELLINGTON',
            'GRANT', 'ARCHDUKE CHARLES', 'TURENNE', 'JACKSON', 'WASHINGTON']

gen_df = bel_merged[bel_merged['co_clean'].isin(generals)].copy()
gen_df['win']         = (gen_df['ach'] >= 6).astype(int)
gen_df['is_underdog'] = (gen_df['force_ratio'] < 1.0).astype(int)

# ── Bayesian win rate ─────────────────────────────────────────────────────────
# Prior: Beta(3.25, 1.75) — weakly informative, equivalent to ~5 battles at 65%
# Posterior: Beta(alpha_prior + wins, beta_prior + losses)
ALPHA_PRIOR = 3.25
BETA_PRIOR  = 1.75

def bayesian_wr(wins, n):
    a = ALPHA_PRIOR + wins
    b = BETA_PRIOR  + (n - wins)
    mean    = a / (a + b)
    lo, hi  = stats.beta.interval(0.95, a, b)
    return mean, lo, hi

# Build summary table
rows = []
for g in generals:
    sub  = gen_df[gen_df['co_clean'] == g]
    n    = len(sub)
    wins = sub['win'].sum()
    raw  = wins / n
    bwr, lo, hi = bayesian_wr(wins, n)
    rows.append({
        'general': g,
        'n':       n,
        'wins':    wins,
        'raw_wr':  raw,
        'bayes_wr':bwr,
        'ci_lo':   lo,
        'ci_hi':   hi,
        'avg_ach': sub['ach'].mean(),
        'intensity': sub['casualty_intensity'].mean(),
        'underdog_pct': sub['is_underdog'].mean() * 100,
        'isNapoleon': g == 'NAPOLEON I',
    })
summary = pd.DataFrame(rows)

# ── Style config ──────────────────────────────────────────────────────────────
NAPOLEON_COLOR = "#E8C060"
GOLD_DIM       = "#A07820"
GOLD_BRIGHT    = "#FFD980"
OTHER_COLOR    = "#7A8FA8"
CI_COLOR       = "#4A6070"
RED            = "#E04040"
INK            = "#0A0804"
INK_MID        = "#161208"
INK_LIGHT      = "#221C10"
CREAM          = "#F5EDD8"
TEXT_DIM       = "#9A8B72"

plt.rcParams.update({
    'figure.facecolor': INK,
    'axes.facecolor':   INK_MID,
    'axes.edgecolor':   '#2A2218',
    'axes.labelcolor':  CREAM,
    'xtick.color':      TEXT_DIM,
    'ytick.color':      TEXT_DIM,
    'text.color':       CREAM,
    'grid.color':       '#2A2218',
    'grid.linestyle':   '--',
    'grid.alpha':       0.5,
    'font.family':      'serif',
})


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — Bayesian win rate with 95% CI
# ─────────────────────────────────────────────────────────────────────────────
plot_df = summary.sort_values('bayes_wr', ascending=False)

fig, ax = plt.subplots(figsize=(13, 7))
x = np.arange(len(plot_df))

colors = [NAPOLEON_COLOR if g else OTHER_COLOR for g in plot_df['isNapoleon']]
bars   = ax.bar(x, plot_df['bayes_wr'] * 100, color=colors, alpha=0.9,
                edgecolor='none', zorder=2, width=0.55)

# 95% CI error bars
ci_lo  = (plot_df['bayes_wr'] - plot_df['ci_lo']) * 100
ci_hi  = (plot_df['ci_hi'] - plot_df['bayes_wr']) * 100
ax.errorbar(x, plot_df['bayes_wr'] * 100,
            yerr=[ci_lo, ci_hi],
            fmt='none', color=CREAM, alpha=0.5,
            capsize=5, capthick=1.2, elinewidth=1.2, zorder=3)

# Raw win rate dots
ax.scatter(x, plot_df['raw_wr'] * 100, color=CREAM, s=28, zorder=4,
           alpha=0.6, label='Raw win rate')

# Value labels
for i, row in enumerate(plot_df.itertuples()):
    ax.text(i, row.bayes_wr * 100 + ci_hi.iloc[i] + 2,
            f'{row.bayes_wr*100:.1f}%\n(n={row.n})',
            ha='center', va='bottom', fontsize=7.5,
            color=NAPOLEON_COLOR if row.isNapoleon else TEXT_DIM)

ax.set_xticks(x)
ax.set_xticklabels(plot_df['general'], rotation=30, ha='right', fontsize=9)
ax.set_ylabel('Win Rate %', fontsize=10)
ax.set_ylim(0, 115)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
ax.set_title('Bayesian Win Rate with 95% Credible Interval',
             fontsize=13, fontweight='bold', color=CREAM, pad=14)
ax.axhline(50, color=TEXT_DIM, linewidth=0.6, linestyle='--', alpha=0.5)
ax.grid(axis='y')
ax.spines[['top', 'right']].set_visible(False)

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    plt.Rectangle((0,0),1,1, color=OTHER_COLOR, alpha=0.9, label='Bayesian win rate'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor=CREAM,
           alpha=0.6, markersize=6, label='Raw win rate'),
    Line2D([0],[0], color=CREAM, alpha=0.5, linewidth=1.5, label='95% credible interval'),
]
ax.legend(handles=legend_elements, fontsize=8, framealpha=0.15, loc='upper right')

fig.tight_layout()
fig.savefig('./BattleML/data/viz_bayesian_winrate.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: viz_bayesian_winrate.png")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 — Underdog vs Favored win rate
# ─────────────────────────────────────────────────────────────────────────────
underdog_rows = []
for g in generals:
    sub = gen_df[gen_df['co_clean'] == g]
    dog = sub[sub['is_underdog'] == 1]
    fav = sub[sub['is_underdog'] == 0]

    fav_bwr, fav_lo, fav_hi = bayesian_wr(fav['win'].sum(), len(fav)) if len(fav) > 0 else (np.nan, np.nan, np.nan)
    dog_bwr, dog_lo, dog_hi = bayesian_wr(dog['win'].sum(), len(dog)) if len(dog) > 0 else (np.nan, np.nan, np.nan)

    underdog_rows.append({
        'general':   g,
        'fav_bwr':   fav_bwr, 'fav_lo': fav_lo, 'fav_hi': fav_hi, 'fav_n': len(fav),
        'dog_bwr':   dog_bwr, 'dog_lo': dog_lo, 'dog_hi': dog_hi, 'dog_n': len(dog),
        'isNapoleon': g == 'NAPOLEON I',
    })
ud_df = pd.DataFrame(underdog_rows).sort_values('fav_bwr', ascending=False)

fig, ax = plt.subplots(figsize=(13, 7))
x = np.arange(len(ud_df))
w = 0.32

colors_fav = [NAPOLEON_COLOR if g else OTHER_COLOR for g in ud_df['isNapoleon']]
colors_dog = [GOLD_DIM      if g else CI_COLOR     for g in ud_df['isNapoleon']]

ax.bar(x - w/2, ud_df['fav_bwr'] * 100, width=w, color=colors_fav,
       alpha=0.9, label='Favored (Bayesian)', edgecolor='none')
ax.bar(x + w/2, ud_df['dog_bwr'].fillna(0) * 100, width=w, color=colors_dog,
       alpha=0.75, label='Underdog (Bayesian)', edgecolor='none', hatch='///')

# CI error bars — favored
fav_lo_err = (ud_df['fav_bwr'] - ud_df['fav_lo']) * 100
fav_hi_err = (ud_df['fav_hi'] - ud_df['fav_bwr']) * 100
ax.errorbar(x - w/2, ud_df['fav_bwr'] * 100,
            yerr=[fav_lo_err, fav_hi_err],
            fmt='none', color=CREAM, alpha=0.4, capsize=4, elinewidth=1)

# n labels
for i, row in enumerate(ud_df.itertuples()):
    ax.text(i - w/2, row.fav_bwr * 100 + fav_hi_err.iloc[i] + 1.5,
            f'n={row.fav_n}', ha='center', va='bottom', fontsize=7, color=TEXT_DIM)
    if row.dog_n > 0 and not np.isnan(row.dog_bwr):
        ax.text(i + w/2, row.dog_bwr * 100 + 1.5,
                f'n={row.dog_n}', ha='center', va='bottom', fontsize=7, color=TEXT_DIM)

ax.set_xticks(x)
ax.set_xticklabels(ud_df['general'], rotation=30, ha='right', fontsize=9)
ax.set_ylabel('Bayesian Win Rate %', fontsize=10)
ax.set_ylim(0, 115)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
ax.set_title('Favored vs Underdog Win Rate (Bayesian)', fontsize=13,
             fontweight='bold', color=CREAM, pad=14)
ax.axhline(50, color=TEXT_DIM, linewidth=0.6, linestyle='--', alpha=0.5)
ax.legend(fontsize=9, framealpha=0.15, loc='upper right')
ax.grid(axis='y')
ax.spines[['top', 'right']].set_visible(False)

fig.tight_layout()
fig.savefig('./BattleML/data/viz_underdog_bayesian.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: viz_underdog_bayesian.png")


# ── Print summary ─────────────────────────────────────────────────────────────
print("\n── Bayesian Win Rate Summary ──")
print(f"{'General':22s} | {'n':>3} | {'Raw WR':>6} | {'Bayes WR':>8} | 95% CI")
print("-" * 70)
for _, row in summary.sort_values('bayes_wr', ascending=False).iterrows():
    print(f"{row['general']:22s} | {int(row['n']):>3} | "
          f"{row['raw_wr']:.3f}  | {row['bayes_wr']:.3f}    | "
          f"[{row['ci_lo']:.3f}, {row['ci_hi']:.3f}]")