import pyblp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


pyblp.options.digits = 3
pyblp.options.verbose = False
pd.options.display.precision = 3
pd.options.display.max_columns = 50


# q1

    ## download data
product = pd.read_csv("products.csv")
product.sample(100)


## descriptives
product.describe()


# q2

# create mkt size
product["market_size"] = product["city_population"] * 90

# create mkt share
product["market_share"] = product["servings_sold"] / product["market_size"]

# get outside share
product["sum_market_share"] = product.groupby('market')['market_share'].transform('sum')
product.sample(100)

product['outside_share'] = 1 - product['sum_market_share']
product['outside_share'].describe()
product.sample(100)

# q3

# create logit_delta
product['logit_delta'] = np.log(product['market_share']) - np.log(product['outside_share'])
product.sample(100)

res = smf.ols(formula = 'logit_delta ~ price_per_serving + mushy', data = product).fit(cov_type = 'HC0')
print(res.summary())
print((1/(7.48))*0.0748)

# q4

# rename variables for pyblp
product = product.rename(columns = {'market': 'market_ids', 'product': 'product_ids', 'market_share': 'shares', 'price_per_serving': 'prices'})

# create prices and instrument
product['demand_instruments0'] = product['prices']

# create problem
ols_problem = pyblp.Problem(pyblp.Formulation('1 + mushy + prices'), product)
print(ols_problem)

# run ols
ols_results = ols_problem.solve(method = '1s')
print(ols_results)

# q5

ols_problem = pyblp.Problem(pyblp.Formulation('1 + prices', absorb = 'C(market_ids) + C(product_ids)'), product)
print(ols_problem)
ols_results = ols_problem.solve(method = '1s')
print(ols_results)

# q6

## first stage
first = smf.ols(formula = 'prices ~ price_instrument + C(market_ids) + C(product_ids)', data = product).fit(cov_type = 'HC0')
print(first.summary())

product['demand_instruments0'] = product['price_instrument']
ols_problem = pyblp.Problem(pyblp.Formulation('1 + prices', absorb = 'C(market_ids) + C(product_ids)'), product)
print(ols_problem)
ols_results = ols_problem.solve(method = '1s')
print(ols_results)

# q7

# counterfactual
counterfactual_data = product[product['market_ids'] == 'C01Q2']
counterfactual_data

# new prices where F1B04 is halved using loc
counterfactual_data.loc[:, 'new_prices'] = counterfactual_data['prices']
counterfactual_data

counterfactual_data.loc[counterfactual_data['product_ids'] == 'F1B04', 'new_prices'] *= 0.5
counterfactual_data

# compute shares
new_shares = ols_results.compute_shares(market_id = 'C01Q2', prices = counterfactual_data['new_prices'])
new_shares
counterfactual_data['new_shares'] = new_shares
counterfactual_data

# q8
## compute elasticities
elasticities = ols_results.compute_elasticities(market_id = 'C01Q2')
plt.colorbar(plt.matshow(elasticities))

## supplemental question

# compute confidence intervals for counterfactual
bootstrap_results = ols_results.bootstrap(draws = 100, seed = 123)

# you need to pass a prices argument with prices replicated along a new axis by as many draws as you have
bootstrap_shares = bootstrap_results.compute_shares(market_id = 'C01Q2', prices = np.tile(counterfactual_data['new_prices'], (100, 1)))
bootstrap_shares

# once you have some bootstrapped shares, compute the same percent changes, and compute the 2.5th and 97.5th percentiles of these changes for each product

counterfactual_data['percent_change_in_shares'] = (bootstrap_shares - counterfactual_data['shares']) / counterfactual_data['shares']
counterfactual_data


## actual solutions
# q1
product_data = pd.read_csv("products.csv")
product_data.sample(n=5, random_state=0)


# q2
product_data['market_size'] = product_data['city_population'] * 90
product_data['market_share'] = product_data['servings_sold'] / product_data['market_size']
product_data['outside_share'] = 1 - product_data.groupby('market')['market_share'].transform('sum')
product_data[['market_share', 'outside_share']].describe()

# q3
product_data['logit_delta'] = np.log(product_data['market_share'] / product_data['outside_share'])
statsmodels_ols = smf.ols('logit_delta ~ 1 + mushy + price_per_serving', product_data)
statsmodels_results = statsmodels_ols.fit(cov_type='HC0')
statsmodels_results.summary2().tables[1]

# q4
product_data = product_data.rename(columns={
    'market': 'market_ids',
    'product': 'product_ids',
    'market_share': 'shares',
    'price_per_serving': 'prices',
})
product_data['demand_instruments0'] = product_data['prices']
ols_problem = pyblp.Problem(pyblp.Formulation('1 + mushy + prices'), product_data)
ols_problem

pd.DataFrame(ols_problem.products.ZD).sample(n=5, random_state=0)

ols_results = ols_problem.solve(method='1s')
ols_results

pd.DataFrame(index=ols_results.beta_labels, data={
    ("Estimates", "Statsmodels"): statsmodels_results.params.values,
    ("Estimates", "PyBLP"): ols_results.beta.flat,
    ("SEs", "Statsmodels"): statsmodels_results.bse.values,
    ("SEs", "PyBLP"): ols_results.beta_se.flat,
})

# q5
fe_problem = pyblp.Problem(pyblp.Formulation('0 + prices', absorb='C(market_ids) + C(product_ids)'), product_data)
fe_problem

# q6
first_stage = smf.ols('prices ~ 0 + price_instrument + C(market_ids) + C(product_ids)', product_data)
first_stage_results = first_stage.fit(cov_type='HC0')
first_stage_results.summary2().tables[1].sort_index(ascending=False)

product_data = product_data.drop(columns='demand_instruments0').rename(columns={'price_instrument': 'demand_instruments0'})
iv_problem = pyblp.Problem(pyblp.Formulation('0 + prices', absorb='C(market_ids) + C(product_ids)'), product_data)
iv_problem

iv_results = iv_problem.solve(method='1s')
iv_results

pd.DataFrame(index=fe_results.beta_labels, data={
    ("Estimates", "OLS"): ols_results.beta[-1:].flat,
    ("Estimates", "+FE"): fe_results.beta.flat,
    ("Estimates", "+IV"): iv_results.beta.flat,
    ("SEs", "OLS"): ols_results.beta_se[-1:].flat,
    ("SEs", "+FE"): fe_results.beta_se.flat,
    ("SEs", "+IV"): iv_results.beta_se.flat,
})

# q7
counterfactual_market = 'C01Q2'
counterfactual_data = product_data.loc[product_data['market_ids'] == counterfactual_market, ['product_ids', 'mushy', 'prices', 'shares']]
counterfactual_data

counterfactual_data['new_prices'] = counterfactual_data['prices']
counterfactual_data.loc[counterfactual_data['product_ids'] == 'F1B04', 'new_prices'] /= 2
counterfactual_data['new_shares'] = iv_results.compute_shares(market_id=counterfactual_market, prices=counterfactual_data['new_prices'])
counterfactual_data['iv_change'] = 100 * (counterfactual_data['new_shares'] - counterfactual_data['shares']) / counterfactual_data['shares']
counterfactual_data

# q8
iv_elasticities = iv_results.compute_elasticities(market_id=counterfactual_market)
pd.DataFrame(iv_elasticities)

## supplemental questions
# q1
product_data['clustering_ids'] = product_data['product_ids']
cluster_problem = pyblp.Problem(pyblp.Formulation('0 + prices', absorb='C(market_ids) + C(product_ids)'), product_data)

cluster_results = cluster_problem.solve(method='1s', se_type='clustered')
pd.DataFrame(index=fe_results.beta_labels, data={
    ("Estimates", "Unclustered"): iv_results.beta.flat,
    ("SEs", "Unclustered"): iv_results.beta_se.flat,
    ("Estimates", "Clustered"): cluster_results.beta.flat,
    ("SEs", "Clustered"): cluster_results.beta_se.flat,
})

# q2
bootstrap_results = cluster_results.bootstrap(draws=100, seed=0)
bootstrap_results

bootstrap_shares = bootstrap_results.bootstrapped_shares[:, product_data['market_ids'] == counterfactual_market]
bootstrap_shares.shape

bootstrap_new_prices = np.tile(counterfactual_data['new_prices'].values, (100, 1))
bootstrap_new_shares = bootstrap_results.compute_shares(market_id=counterfactual_market, prices=bootstrap_new_prices)
bootstrap_changes = 100 * (bootstrap_new_shares - bootstrap_shares) / bootstrap_shares

counterfactual_data['iv_change_lb'] = np.squeeze(np.percentile(bootstrap_changes, 2.5, axis=0))
counterfactual_data['iv_change_ub'] = np.squeeze(np.percentile(bootstrap_changes, 97.5, axis=0))
counterfactual_data
