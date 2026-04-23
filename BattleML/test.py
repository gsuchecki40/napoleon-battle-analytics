import pandas as pd
import numpy as np
from scipy import stats

bel = pd.read_csv('./BattleML/CDB90/data/belligerents.csv')
df  = pd.read_csv('./BattleML/data/battles_clustered.csv')

bel['co_clean'] = bel['co'].replace({
    'BONAPARTE':             'NAPOLEON I',
    'WELLINGTON & BLUECHER': 'WELLINGTON',
})

bel_merged = bel.merge(df[['isqno', 'kmeans']], on='isqno', how='left')

generals = ['NAPOLEON I', 'FREDERICK II', 'LEE', 'WELLINGTON',
            'GRANT', 'ARCHDUKE CHARLES', 'TURENNE', 'JACKSON', 'WASHINGTON']

gen_df = bel_merged[bel_merged['co_clean'].isin(generals)].copy()
gen_df['win'] = (gen_df['ach'] >= 6).astype(int)

# Prior: Beta(alpha, beta) fitted to overall win rates across all generals

alpha_prior = 3.25
beta_prior  = 1.75

print("General            | n  | Wins | Raw WR | Bayes WR | 95% CI")
print("-" * 70)
for g in generals:
    sub  = gen_df[gen_df['co_clean'] == g]
    n    = len(sub)
    wins = sub['win'].sum()
    raw  = wins / n

    # Posterior Beta(alpha_prior + wins, beta_prior + losses)
    a_post = alpha_prior + wins
    b_post = beta_prior  + (n - wins)
    bayes  = a_post / (a_post + b_post)
    lo, hi = stats.beta.interval(0.95, a_post, b_post)

    print(f"{g:22s} | {n:2d} | {wins:4.0f} | {raw:.3f}  | {bayes:.3f}    | [{lo:.3f}, {hi:.3f}]")