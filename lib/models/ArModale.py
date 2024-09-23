from sklearn.linear_model import LinearRegression

from lib.models.AForecastModel import AForecastModel


class ArModel(AForecastModel):
    def __init__(self, degree: int):
        super().__init__(f'AR({degree})')
        self.degree = degree
        self.model = LinearRegression(fit_intercept=False)

    def fit(self, X, Y):
        x_model = X[:, :self.degree]
        self.model = self.model.fit(x_model, Y)
        return self

    def predict(self, X):
        x_model = X[:, :self.degree]
        return self.model.predict(x_model)

    def force_coef(self, coef):
        self.model.fit(X=[[0]*self.degree], y=[0])
        self.model.coef_ = coef.reshape(1, self.degree)
        return self

