import numpy as np

class LinearRegression:
    def fit(self, _X, _y): 
        # _X nesse caso é um vetor de entrada, mas poderia ser uma matriz, _y são os labels, nesse caso salários
        # Adiciona um vetor de 1s na primeira coluna de _X para configurar o intercepto (termo de bias)
        
        _X = np.array(_X)
        _y = np.array(_y)

        # Calcula os coeficientes da regressão linear
        _Xtranspose = _X.T
        _Xtranspose_dot_X = _Xtranspose.dot(_X)
        _Xtranspose_dot_X_inv = np.linalg.inv(_Xtranspose_dot_X)
        _Xtranspose_dot_y = _Xtranspose.dot(_y)

        self.w = _Xtranspose_dot_X_inv.dot(_Xtranspose_dot_y)  # Vetor de pesos (coeficientes)


    def predict(self, _x):
        X = np.array(_x)
        return X.dot(self.w)  # Predição em si
    
    def getW(self):
        return self.w