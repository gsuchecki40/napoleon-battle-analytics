import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

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
gen_df['win']        = (gen_df['ach'] >= 6).astype(int)
gen_df['is_underdog'] = (gen_df['force_ratio'] < 1.0).astype(int)

NAPOLEON_COLOR = "#E8C060"
OTHER_COLOR    = "#556070"
OTHER_BRIGHT   = "#7A8FA8"
RED            = "#E04040"
INK            = "#0A0804"
INK_MID        = "#161208"
INK_LIGHT      = "#221C10"
CREAM          = "#F5EDD8"
TEXT_DIM       = "#9A8B72"

plt.rcParams.update({
    'figure.facecolor':  INK,
    'axes.facecolor':    INK_MID,
    'axes.edgecolor':    '#2A2218',
    'axes.labelcolor':   CREAM,
    'xtick.color':       TEXT_DIM,
    'ytick.color':       TEXT_DIM,
    'text.color':        CREAM,
    'grid.color':        '#2A2218',
    'grid.linestyle':    '--',
    'grid.alpha':        0.5,
    'font.family':       'serif',
})


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — Underdog vs Favored win rate per general
# ─────────────────────────────────────────────────────────────────────────────
underdog_data = []
for g in generals:
    sub = gen_df[gen_df['co_clean'] == g]
    dog = sub[sub['is_underdog'] == 1]
    fav = sub[sub['is_underdog'] == 0]
    underdog_data.append({
        'general':      g,
        'underdog_wr':  dog['win'].mean() if len(dog) > 0 else np.nan,
        'underdog_n':   len(dog),
        'favored_wr':   fav['win'].mean() if len(fav) > 0 else np.nan,
        'favored_n':    len(fav),
    })
ud_df = pd.DataFrame(underdog_data).sort_values('favored_wr', ascending=False)

fig, ax = plt.subplots(figsize=(13, 7))
x = np.arange(len(ud_df))
w = 0.35

bars_fav = ax.bar(x - w/2, ud_df['favored_wr'] * 100,  width=w,
                  color=[NAPOLEON_COLOR if g == 'NAPOLEON I' else OTHER_BRIGHT for g in ud_df['general']],
                  alpha=0.9, label='Favored (force ratio ≥ 1.0)', edgecolor='none')

bars_dog = ax.bar(x + w/2, ud_df['underdog_wr'] * 100, width=w,
                  color=[NAPOLEON_COLOR if g == 'NAPOLEON I' else OTHER_COLOR for g in ud_df['general']],
                  alpha=0.6, label='Underdog (force ratio < 1.0)', edgecolor='none',
                  hatch='///')

# Sample size labels
for i, row in enumerate(ud_df.itertuples()):
    if not np.isnan(row.favored_wr):
        ax.text(i - w/2, row.favored_wr * 100 + 1.5, f'n={row.favored_n}',
                ha='center', va='bottom', fontsize=7, color=TEXT_DIM)
    if not np.isnan(row.underdog_wr):
        ax.text(i + w/2, row.underdog_wr * 100 + 1.5, f'n={row.underdog_n}',
                ha='center', va='bottom', fontsize=7, color=TEXT_DIM)

ax.set_xticks(x)
ax.set_xticklabels(ud_df['general'], rotation=30, ha='right', fontsize=9)
ax.set_ylabel('Win Rate %', fontsize=10)
ax.set_ylim(0, 115)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
ax.set_title('Win Rate: Favored vs Underdog by General', fontsize=13, fontweight='bold',
             color=CREAM, pad=14)
ax.legend(fontsize=9, framealpha=0.2, loc='upper right')
ax.axhline(50, color=TEXT_DIM, linewidth=0.6, linestyle='--', alpha=0.5)
ax.grid(axis='y')
ax.spines[['top', 'right']].set_visible(False)

fig.tight_layout()
fig.savefig('./BattleML/data/viz_underdog_winrate.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: viz_underdog_winrate.png")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 — Cluster win rate: Napoleon vs peers (only Napoleon's clusters)
# ─────────────────────────────────────────────────────────────────────────────
nap_clusters = sorted(gen_df[gen_df['co_clean'] == 'NAPOLEON I']['kmeans'].unique())
cluster_wr   = (
    gen_df[gen_df['kmeans'].isin(nap_clusters)]
    .groupby(['kmeans', 'co_clean'])['win']
    .agg(['mean', 'count'])
    .reset_index()
)
cluster_wr.columns = ['kmeans', 'general', 'win_rate', 'n']
cluster_wr = cluster_wr[cluster_wr['n'] >= 2]   # drop tiny samples
cluster_wr['win_rate'] *= 100

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
axes = axes.flatten()

for ax_idx, c in enumerate(nap_clusters):
    ax = axes[ax_idx]
    sub = cluster_wr[cluster_wr['kmeans'] == c].sort_values('win_rate', ascending=False)

    colors = [NAPOLEON_COLOR if g == 'NAPOLEON I' else OTHER_BRIGHT for g in sub['general']]
    bars   = ax.bar(range(len(sub)), sub['win_rate'], color=colors, edgecolor='none', alpha=0.9)

    # Napoleon highlight line
    nap_row = sub[sub['general'] == 'NAPOLEON I']
    if len(nap_row):
        ax.axhline(nap_row['win_rate'].values[0], color=NAPOLEON_COLOR,
                   linewidth=1, linestyle='--', alpha=0.4)

    for bar, (_, row) in zip(bars, sub.iterrows()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{row['win_rate']:.0f}%\n(n={int(row['n'])})",
                ha='center', va='bottom', fontsize=7.5, color=CREAM)

    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels(sub['general'], rotation=35, ha='right', fontsize=8)
    ax.set_ylim(0, 120)
    ax.set_ylabel('Win Rate %', fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
    ax.set_title(cluster_names[c], fontsize=10, fontweight='bold', color=NAPOLEON_COLOR
                 if c in [0, 1, 5] else CREAM, pad=8)
    ax.axhline(50, color=TEXT_DIM, linewidth=0.5, linestyle=':', alpha=0.5)
    ax.grid(axis='y')
    ax.spines[['top', 'right']].set_visible(False)

fig.suptitle("Napoleon's Win Rate vs Peers — By Battle Type",
             fontsize=14, fontweight='bold', color=CREAM, y=1.01)
fig.tight_layout()
fig.savefig('./BattleML/data/viz_cluster_winrate_peers.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: viz_cluster_winrate_peers.png")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3 — Achievement score distribution per general
# ─────────────────────────────────────────────────────────────────────────────
ach_dist = (
    gen_df.groupby(['co_clean', 'ach'])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=range(0, 11), fill_value=0)
)

# Normalize to % of each general's battles
ach_pct = ach_dist.div(ach_dist.sum(axis=1), axis=0) * 100

# Sort generals: Napoleon first, then by avg ach
gen_order = ['NAPOLEON I'] + [g for g in generals if g != 'NAPOLEON I'
                               and g in ach_pct.index]
ach_pct = ach_pct.loc[gen_order]

fig, ax = plt.subplots(figsize=(14, 8))

score_cols = [c for c in ach_pct.columns if ach_pct[c].sum() > 0]
x          = np.arange(len(score_cols))
n_generals = len(gen_order)
bar_w      = 0.8 / n_generals

for i, g in enumerate(gen_order):
    vals   = [ach_pct.loc[g, c] if c in ach_pct.columns else 0 for c in score_cols]
    color  = NAPOLEON_COLOR if g == 'NAPOLEON I' else OTHER_BRIGHT
    alpha  = 1.0 if g == 'NAPOLEON I' else 0.55
    zorder = 3 if g == 'NAPOLEON I' else 2
    offset = (i - n_generals / 2 + 0.5) * bar_w
    ax.bar(x + offset, vals, width=bar_w, color=color, alpha=alpha,
           label=g, edgecolor='none', zorder=zorder)

ax.axvline(x[score_cols.index(5)] if 5 in score_cols else 2,
           color=TEXT_DIM, linewidth=0.8, linestyle='--', alpha=0.5,
           label='Stalemate threshold (5)')

ax.set_xticks(x)
ax.set_xticklabels([f'ACH {c}' for c in score_cols], fontsize=9)
ax.set_ylabel('% of Battles', fontsize=10)
ax.set_title('Achievement Score Distribution — Napoleon vs Peers',
             fontsize=13, fontweight='bold', color=CREAM, pad=14)
ax.legend(fontsize=7.5, framealpha=0.15, ncol=3, loc='upper left')
ax.grid(axis='y')
ax.spines[['top', 'right']].set_visible(False)

# Annotation
ax.annotate('Napoleon stacks 7s, 8s, 9s.\nPeers cluster at 5–6.',
            xy=(x[score_cols.index(7)] if 7 in score_cols else 4, 28),
            xytext=(x[score_cols.index(9)] if 9 in score_cols else 6, 35),
            arrowprops=dict(arrowstyle='->', color=NAPOLEON_COLOR, lw=1.2),
            fontsize=9, color=NAPOLEON_COLOR, fontstyle='italic')

fig.tight_layout()
fig.savefig('./BattleML/data/viz_ach_distribution.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: viz_ach_distribution.png")


# ── Print summary tables ──────────────────────────────────────────────────────
print("\n── Underdog vs Favored Win Rates ──")
print(ud_df.to_string(index=False))

print("\n── Cluster Win Rates (Napoleon's clusters, n>=2) ──")
print(cluster_wr.pivot(index='general', columns='kmeans', values='win_rate').round(1).to_string())

print("\n── ACH Distribution (%) ──")
print(ach_pct.round(1).to_string())