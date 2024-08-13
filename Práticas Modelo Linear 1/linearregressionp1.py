import numpy as np

# esse arquivo de reg linear é utilizado para quando existe mais de uma feature (mais de uma coluna)


class LinearRegression:
    def fit_lr(self, _X, _y): 
        # Adicionar uma coluna de 1s à matriz _X para incluir o termo de bias
        _X = np.hstack((np.ones((len(_X), 1)), _X)) 
        _y = np.array(_y)

        # Calcula os coeficientes da regressão linear
        _Xtranspose = _X.T
        _Xtranspose_dot_X = _Xtranspose.dot(_X)
        _Xtranspose_dot_X_inv = np.linalg.inv(_Xtranspose_dot_X)
        _Xtranspose_dot_y = _Xtranspose.dot(_y)

        self.w = _Xtranspose_dot_X_inv.dot(_Xtranspose_dot_y)  # Vetor de pesos (coeficientes)

    def predict_lr(self, _x):
        # Adicionar uma coluna de 1s para o termo de bias se _x for uma matriz
        if len(_x.shape) == 1:
            _x = np.hstack(([1], _x))
        else:
            _x = np.hstack((np.ones((_x.shape[0], 1)), _x))
        
        return _x.dot(self.w)  # Predição em si
    
    def getW_lr(self):
        return self.w