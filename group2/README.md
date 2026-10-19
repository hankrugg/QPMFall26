Group 2: Andrew Rubin, Aidan Altorelli, Ian Schunk, Marilyn Risinger

The notebook Group2_StockFactorBetas.ipynb runs the regressions for every stock on stagger 2, using the time-weighted returns and residual sector factors from companies_and_staggered_returns.xlsx. A foreign stock is regressed on its own currency first and that beta is kept (US stocks skip this). The residual is then regressed on the other currencies. Each round, the currency whose t-stat is closest to zero is dropped if that t-stat is between -2 and 2, and the regression is refit, until every remaining currency is outside that band or none are left. The residual is then regressed on SPY and that beta is kept. The same two-step rule is used for the 11 sector factors, and every stock keeps its own sector beta. Whatever is left is the stock's residual return.

The results are saved in group2_stock_betas.xlsx. The "Betas" tab has one row per stock. A dropped factor has a beta of 0. The "Residuals" tab has the residual returns. The "Covariance" tab is the 43 by 43 covariance matrix of those residuals, and it is symmetric.

A few things to point out:

- The residuals subtract both alpha and beta at each step, matching how group0 computed the sector residuals.

- AUD is still the currency beta kept most often, for 21 of 43 stocks. CAD is kept for 7 of 43.

- The CAD/USD series in the input file is the corrected series.
