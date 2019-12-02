import math
import pandas as ps
import matplotlib
import matplotlib.pyplot as mp
import numpy as np
import statsmodels
import statsmodels.api as sd
import statsmodels.formula.api as sf
data = ps.read_csv('dataset.csv')
ydata = data.query('year == 1985')
size = 1e-6 * ydata.population

colors = ydata.region.map({'Africa': 'skyblue', 'Europe': 'gold', 'America': 'palegreen', 'Asia': 'coral'})

def plotdata():
    gdata.plot.scatter('age6_surviving','babies_per_woman',
                      c=colors,s=size,linewidths=0.6,edgecolor='k',alpha=0.6)
plotdata()
mod=sf.ols(formula='babies_per_woman ~ 1',data=ydata)
grdmean=mod.fit()
grdmean
def plotfit(fit):
    plotdata()
    mp.scatter(ydata.age6_surviving,fit.predict(ydata),
               c=colors,s=30,linewidths=0.6,edgecolor='k',marker='D')
plotfit(grdmean)
grdmean.params
ydata.babies_per_woman.mean()
gpmean = sf.ols(formula='babies_per_woman ~ -1 + region', data=ydata).fit()
plotfit(gpmean)
gpmean.params
ydata.groupby('region').babies_per_woman.mean()
survive = sf.ols(formula='babies_per_woman ~ -1 + region + age6_surviving', data=ydata).fit()
sur_by_region_population = sf.ols(
    formula='babies_per_woman ~ -1 + region + age6_surviving'
            '+ age6_surviving:region - age6_surviving + population',
    data=ydata).fit()
plotfit(survive)
survive.params
plotfit(sur_by_region_population)
sur_by_region_population.params
mp.scatter(ydata.age6_surviving,gpmean.resid)
mp.scatter(ydata.age6_surviving,survive.resid)
mp.scatter(ydata.age6_surviving,sur_by_region_population.resid)
for model in [gpmean,survive,sur_by_region_population]:
    print(model.mse_resid)
for model in [gpmean,survive,sur_by_region_population]:
    print(model.rsquared)
for model in [gpmean,survive,sur_by_region_population]:
    print(model.rsquared)
for model in [gpmean,survive,sur_by_region_population]:
    print(model.fvalue)
survive.summary()
sd.stats.anova_lm(sur_by_region_population)
sd.stats.anova_lm(survive)
sd.stats.anova_lm(gpmean)

