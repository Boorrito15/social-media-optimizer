# alden.ipynb — run results (text-only features, 5-fold grouped CV, 11,306 posts)

## Regression metric (rob.ipynb): predict raw (engagement, views), MAE in raw engagement units vs median median-baseline.

- KNN:  MAE(raw engagement) = 14,470 ±794 | median baseline = 12,427 | -2,044 (-16%) — LOSES to baseline
- LightGBM: not run here (needs libomp — run in user's Jupyter)
- RandomForest: too slow to complete from the shell (100 trees × 5 folds on 1,554 sparse cols)

## Classification metric (rob.ipynb cells 45-66): bin views & engagement into 2x2 grid (N_BINS=2), accuracy vs majority baseline.

Bin distribution: views 0=7074/1=4232; engagement 0=7201/1=4105
Grid labels: v0_e0=6591, v1_e1=3622, v1_e0=610, v0_e1=483

- KNN:         views_acc=0.782(base 0.626) | eng_acc=0.797(base 0.637) | joint_acc=0.715(base 0.583) | +0.132 ✓ BEATS baseline
- RandomForest: views_acc=0.786(base 0.626) | eng_acc=0.788(base 0.637) | joint_acc=0.704(base 0.583) | +0.121 ✓ BEATS baseline
- LightGBM: not run here (needs libomp — run in user's Jupyter)

## Conclusion
Classification (predicting the 2x2 engagement/views quadrant) clearly works and beats the
majority baseline (~0.70-0.72 joint accuracy vs 0.58 baseline). The regression metric (raw
MAE) loses to baseline for KNN because raw engagement/views are dominated by platform/audience
size that text features can't recover. This strongly suggests the classification quadrant
target is the more useful success signal for a text-only input.

Note: LightGBM and RandomForest-regression couldn't run in the agent shell (libomp missing /
too slow); run cells 13, 14, 18 in the user's own Jupyter to complete the comparison.
