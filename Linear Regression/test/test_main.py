from pathlib import Path
import polars as pl
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

from src.main import LinearRegresion as CLinearRegression




def get_models():
    path = Path(r"G:\Mi unidad\Desarrollos\Models\Linear Regression\laptop_price - dataset.csv")
    df = pl.read_csv(path)

    y = df["Price (Euro)"]
    X = df.drop("Price (Euro)")
    X = X.with_columns(pl.lit(1).cast(pl.Int64).alias("Intercept"))
    X = X.select(pl.col(pl.Int64,pl.Float64))

    X_scipy = X.to_numpy()
    y_scipy = y.to_numpy()

    linearregresion_custom = CLinearRegression(y,X,intercept = True)
    linearregresion_sp = LinearRegression().fit(X_scipy,y_scipy)
    linearregresion_sm = sm.OLS(y_scipy,X_scipy)
    

    
    return linearregresion_custom, linearregresion_sp, linearregresion_sm



def test_params():
    linearregresion_custom, linearregresion_sp, linearregresion_sm = get_models()
    
    # Scipy
    scipy_params = list(linearregresion_sp.coef_)[:-1] + [linearregresion_sp.intercept_]

    # Custom
    custom_params = linearregresion_custom.params

    pct_dif = 0
    for custom_param,scipy_param in zip(scipy_params,custom_params):
        pct_dif += (scipy_param-custom_param)/scipy_param

    assert pct_dif<0.001


def test_resid():
    linearregresion_custom, linearregresion_sp, linearregresion_sm = get_models()

    resid_custom = linearregresion_custom.resid
    resid_sm = linearregresion_sm.fit().resid

    pct_dif = 0
    for custom,sm in zip(resid_sm,resid_custom):
        pct_dif += np.abs(sm-custom)/sm

    assert pct_dif<0.001

    