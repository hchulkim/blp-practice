import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import pyblp
    import pandas as pd
    import numpy as np
    return np, pd, pyblp


@app.cell
def _(pd):
    product_data = pd.read_csv("input/ps3-q3-data.csv")
    product_data.head()
    return (product_data,)


@app.cell
def _(pd):
    iv_data = pd.read_csv("input/ps3-q3-iv.csv")
    iv_data.head()
    return (iv_data,)


@app.cell
def _(iv_data, pd, product_data):
    # note the colon BEFORE the comma: [:, ...]
    iv_noid = iv_data.loc[:, "haus_iv":"closest_calories"]

    iv_slice = iv_noid.rename(columns = {
        "n_competing":"demand_instruments0",
        "avg_calories":"demand_instruments1",
        "n_organic":"demand_instruments2",
        "closest_calories":"demand_instruments3"
    })

    # align rows before concatenating; safest is to reset both indices
    full_data = pd.concat(
        [
            product_data.rename(columns={
                "price": "prices",
                "mkt_id": "market_ids",
                "product_id": "product_ids",
                "market_share": "shares",
            }).reset_index(drop=True),
            iv_slice.reset_index(drop=True),
        ],
        axis=1
    )
    full_data.head()
    return (full_data,)


@app.cell
def _(pyblp):
    X1_formulation = pyblp.Formulation("0 + prices + organic + calories")
    X2_formulation = pyblp.Formulation("0 + prices + calories")
    product_formulations = (X1_formulation, X2_formulation)
    product_formulations
    return (product_formulations,)


@app.cell
def _(pyblp):
    mc_integration = pyblp.Integration('monte_carlo', size=100, specification_options={'seed':0})
    mc_integration
    return (mc_integration,)


@app.cell
def _(full_data, mc_integration, product_formulations, pyblp):
    mc_problem = pyblp.Problem(product_formulations, full_data, integration=mc_integration)
    mc_problem
    return (mc_problem,)


@app.cell
def _(pyblp):
    bfgs = pyblp.Optimization('bfgs', {'gtol': 1e-12})
    bfgs
    return (bfgs,)


@app.cell
def _(bfgs, mc_problem, np):
    results1 = mc_problem.solve(sigma=np.eye(2), optimization=bfgs)
    results1
    return


if __name__ == "__main__":
    app.run()
