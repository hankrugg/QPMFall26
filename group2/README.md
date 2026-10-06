Group 2: Andrew Rubin, Aidan Altorelli, Ian Schunk, Marilyn Risinger

The notebook Group2_StockFactorBetas.ipynb runs the stepwise regressions for every stock, using the time-weighted returns and residual sector factors from companies_and_staggered_returns.xlsx. For each stock in each stagger: foreign stocks are regressed on their local currency first and the beta is kept (US stocks skip this), then the residuals are regressed on AUD where the beta is kept only if |t| > 2 and set to zero otherwise, then on SPY (beta kept), then on the stock's own residual sector factor (beta kept). Whatever is left after all four steps is the stock-specific return.

The results are saved in group2_stock_betas.xlsx. "Betas by Stagger" has one row per stock per stagger with all the betas, the AUD t-stat, and the residual risk. "Betas XRD Average" averages each stock's betas across the four staggers. The "Residuals stagger_1" through "stagger_4" tabs have the stock-specific returns.

A few things to point out:

- The residuals subtract both alpha and beta at each step, matching how group0 computed the sector residuals.

- The AUD beta passes the |t| > 2 test in 105 of 172 regressions, which is more often than we expected. Could be worth discussing as a group whether that is fine or the AUD step should be handled differently.

- The final residuals are exactly uncorrelated with the sector factors and almost exactly uncorrelated with SPY, but the foreign stocks pick back up some correlation with their local currency (up to about 0.35). This happens because the AUD and SPY steps subtract pieces that are themselves correlated with the local currencies. The verification section of the notebook shows this.
