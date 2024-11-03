import polars as pl
from pathlib import Path
import numpy as np
from numpy.linalg import inv



class LinearRegresion():
    def __init__(self,y,X,intercept):
        if intercept:
            X = X.with_columns(pl.lit(1).cast(pl.Int64).alias("Intercept"))
        self.X = X
        self.y = y
        self.numerical_columns = X.select(pl.col(pl.Int64,pl.Float64))

        self.params_dict = self.get_params()
        self.y_estimated = self.get_estimated_y()
        self.resid = self.y - self.y_estimated


    def get_params(self):
      #  """\hat \beta = (X'X)^{-1}X'Y"""
        y = self.y
        X = self.numerical_columns

        X_columns = X.columns
        X = X.to_numpy()
        X_transpose = np.transpose(X)
        X_mul = np.matmul(X_transpose,X)
        X_mul_inverse = inv(X_mul)
        X_mul_inverse_mul = np.matmul(X_mul_inverse,X_transpose)
        beta = np.matmul(X_mul_inverse_mul,y)
        self.params = beta
        params_dict = dict()
        for col_name,param in zip(X_columns,beta):
            params_dict[col_name] = param


        return params_dict
    
    def get_estimated_y(self):
        X = np.array(self.numerical_columns)
        y_estimated = np.matmul(X,self.params)

        return y_estimated


if __name__ == "__main__":
    
    path = Path(r"G:\Mi unidad\Desarrollos\Models\Linear Regression\laptop_price - dataset.csv")

    df = pl.read_csv(path)

    y = df["Price (Euro)"]
    
    X = df.drop("Price (Euro)")


    linearregresion = LinearRegresion(y,X,intercept = True)


    print(linearregresion.resid)