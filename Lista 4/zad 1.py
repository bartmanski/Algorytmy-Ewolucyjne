
import numpy as np
import scipy
import scipy.linalg
from scipy.stats import chi2
import plotly.graph_objs as go
import plotly.express as px

def draw_ellipse(mean, cov, color, label, alpha=0.95, n_points=100):
    """Draws a 95% confidence ellipse for the distribution N(mean, cov)."""
    # Eigen decomposition of the covariance matrix
    eigen_vals, eigen_vecs = np.linalg.eig(cov)
    # Scaling factors for the ellipse axes
    scale = np.sqrt(chi2.ppf(alpha, df=2)) * np.sqrt(eigen_vals)
    # Generate ellipse points
    t = np.linspace(0, 2 * np.pi, n_points)
    ellipse = np.array([scale[0] * np.cos(t), scale[1] * np.sin(t)])  # Shape (2, n_points)
    print(ellipse.shape)
    print(ellipse)
    print(eigen_vecs.shape)
    # Rotate the ellipse
    rotated_ellipse = eigen_vecs @ ellipse  # Shape (2, n_points)
    # Translate the ellipse
    ellipse = rotated_ellipse.T + mean  # Broadcasting mean (1, 2) to (n_points, 2)
    return go.Scatter(x=ellipse[:, 0], y=ellipse[:, 1], mode='lines', line=dict(color=color), name=label)

def visualize_generation(x, fitvals, xmean, C, next_xmean, next_C,g):
    if(g%5!=0):
        return 0
    """Visualize the generation with points and ellipses."""
    # Sort points by fitness
    sorted_indices = np.argsort(fitvals)
    selected = x[sorted_indices[:len(x) // 2]]  # Selected individuals
    rejected = x[sorted_indices[len(x) // 2:]]  # Rejected individuals

    # Scatter plots for points
    selected_points = go.Scatter(x=selected[:, 0], y=selected[:, 1], mode='markers',
                                  marker=dict(color='green', size=8), name=f'Selected (g={g})')
    rejected_points = go.Scatter(x=rejected[:, 0], y=rejected[:, 1], mode='markers',
                                  marker=dict(color='red', size=8), name=f'Rejected (g={g})')
    all_points = go.Scatter(x=x[:, 0], y=x[:, 1], mode='markers',
                            marker=dict(color='blue', size=6), name=f'All Individuals (g={g})')

    # Ellipses
    ellipse_current = draw_ellipse(xmean, C, 'blue', f'Ellipse (g={g})')
    ellipse_next = draw_ellipse(next_xmean, next_C, 'orange', f'Ellipse (g={g+1})')

    # Layout and plot
    layout = go.Layout(title='elipsy', xaxis=dict(title='x1'), yaxis=dict(title='x2'),
                       showlegend=True)
    fig = go.Figure(data=[all_points, selected_points, rejected_points, ellipse_current, ellipse_next], layout=layout)
    fig.show()

def plot_3d_function(f, a = 10, k = 100):
    x = np.linspace(-a, a, k)
    y = x.copy()
    xy = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
    z = f(xy)
    fig = go.Figure(data=[go.Surface(x = x,  y= y, z=z.reshape((x.shape[0], -1)))])
    fig.update_layout(title = f.__name__, margin=dict(l=65, r=50, b=65, t=90))
    fig.show()

#Funkcje celu
def sphere_function(X):
    return np.sum(X**2, axis=1)
#plot_3d_function(sphere_function)
def ellipsoid_function(X, a=2):
    n = X.shape[-1]
    return np.sum((a**(np.arange(n)/(n-1)))*X**2, axis=1)
#plot_3d_function(ellipsoid_function, 20, 200)
def rastrigin_function(X):
    return 10.0 * X.shape[1] + np.sum(X**2, axis=1) - 10.0 * np.sum(np.cos(2 * np.pi * X), axis=1)
#plot_3d_function(rastrigin_function)
def schwefel_function(X):
    return 418.9829 * X.shape[1] - np.sum(X * np.sin(np.sqrt(np.abs(X))), axis=1)
#plot_3d_function(schwefel_function, 100, 1000)
def griewank_function(X):
    return 1 + np.sum(X**2 / 4000, axis=1) - np.prod(np.cos(X / np.sqrt(np.linspace(1, X.shape[1], X.shape[1]))), axis=1)
#plot_3d_function(griewank_function)
def cigar_function(X, a = 3):
    if len(X.shape) == 1:
        X = X[np.newaxis,:]
    x1 = X[:,0]**2
    x2 = np.sum(a*X[:,1:]**2, axis = 1)
    return x1+x2
#plot_3d_function(cigar_function, 100, 1000)
def discus_function(X, a = 3):
    if len(X.shape) == 1:
        X = X[np.newaxis,:]
    x1 = a*X[:,0]**2
    x2 = np.sum(X[:,1:]**2, axis = 1)
    return x1+x2
#plot_3d_function(discus_function, 100, 100)
def cigar_discus_function(X, a = 2):
    if len(X.shape) == 1:
        X = X[np.newaxis,:]
    x1 = a*X[:,0]**2
    x2 = np.sum((a**0.5)*X[:,1:-1]**2, axis = 1)
    x3 = X[:,-1]**2
    return x1+x2+x3
#plot_3d_function(cigar_discus_function)
def parab_ridge_function(X):
    if len(X.shape) == 1:
        X = X[np.newaxis,:]
    x1 = - X[:,0]
    x2 = 100*np.sum(X[:,1:]**2, axis = 1)
    return x1+x2
#plot_3d_function(parab_ridge_function)
def two_axes_function(X, a = 5):
    if len(X.shape) == 1:
        X = X[np.newaxis,:]
    n = X.shape[-1]
    k = n//2
    x1 = np.sum(a*X[:,:k]**2, axis = 1)
    x2 = np.sum(X[:,k:]**2, axis = 1)
    return x1+x2
#plot_3d_function(two_axes_function, 10)
#Implementacja CMA-ES
class CMA_ES:
    def __init__(self, x0, sigma, maxfevals = 10000, popsize = None, weights = None):
        N = x0.shape[0]
        # rozmiar jednego typeczka
        self.dimension = N
        # 
        self.chiN = N**0.5 * (1 - 1. / (4 * N) + 1. / (21 * N**2))
        # rozmiar populacji dzieci
        self.lam = 4 + int(3 * np.log(N)) if not popsize else popsize
        print(f"Popsize: {self.lam}")
        # rozmiar popu;acji rodzicow
        self.mu = int(self.lam / 2)
        
        if weights:
            self.weights = weights
        else:
            self.weights = np.array([np.log(self.lam / 2 + 0.5) - np.log(i + 1) if i < self.mu else 0
                        for i in range(self.lam)])
            self.weights /= np.sum(self.weights)
        self.mueff = np.sum(self.weights)**2 / np.sum(self.weights**2)
        
        self.cc = (4 + self.mueff/N) / (N+4 + 2 * self.mueff/N)
        self.cs = (self.mueff + 2) / (N + self.mueff + 5)
        self.c1 = 2 / ((N + 1.3)**2 + self.mueff) 
        self.cmu = min([1 - self.c1, 2 * (self.mueff - 2 + 1/self.mueff) / ((N + 2)**2 + self.mueff)])
        self.damps = 2 * self.mueff/self.lam + 0.3 + self.cs

        self.xmean = x0[:]
        self.sigma = sigma
        self.pc = np.zeros(N) 
        self.ps =np.zeros(N) 
        self.lazy_gap_evals = 0.5 * N * self.lam * (self.c1 + self.cmu)**-1 / N**2
        self.maxfevals = maxfevals
        self.C = np.identity(N)
        self.counteval = 0 
        self.fitvals = []   
        self.best = (x0, None)
        self.condition_number = 1
        self.eigen_values = np.ones(N)
        self.eigen_vectors = np.identity(N)
        self.updated_eval = 0
        self.inv_sqrt = np.ones(N)

    def _update_eigensystem(self, current_eval, lazy_gap_evals):
        if current_eval <= self.updated_eval + lazy_gap_evals:
            return self
        self.eigen_values, self.eigen_vectors = np.linalg.eig(self.C)
        self.inv_sqrt = self.eigen_vectors @ np.diag(self.eigen_values**-0.5) @ self.eigen_vectors.T
        self.condition_number = self.eigen_values.max() / self.eigen_values.min()
         
    def sample(self):
        """Wylosuj próbkę nowych osobników"""
        # wzorek z którego korzystamy  x_k^(g+1) ∼ m^(g) + σ^(g) x_k^(g+1) ∼ m^(g) + σ^(g) N(0, C^(g))
        #upewniamy sie że wektory własne są zaktualizowane
        self._update_eigensystem(self.counteval, self.lazy_gap_evals)
        # bierzemy próbki o rozmiarze N (jednego osobnika) i bierzemy ich tyle ile chcemy potomstwa
        z = np.random.randn(self.lam, self.dimension)
        # N(0, C^(g)) 
        y = z @ np.diag(self.eigen_values**0.5) @ self.eigen_vectors.T
        # ostatni krok wzorku mnożymy naszych przedstawicieli rozkladu * sigma - dodajemy do sredniej
        x = self.xmean + self.sigma * y
        return x
    
    def update(self, x, fitvals,g):
        """Zaktualizuj wartości uzyskanych parametrów"""
        self.counteval += fitvals.shape[0] # Zwiększamy licznik wykonań
        N = self.xmean.shape[0]
        x_old = self.xmean.copy()
        C_old = self.C.copy() 
        
        # Posortuj osobniki po wartości funkcji celu
        indices = np.argsort(fitvals)
        x = x[indices]
        self.fitvals = fitvals[indices]
        self.best = (x[0], self.fitvals[0])
        
        self.xmean = (self.weights @ x)
        # to by było korzystając z m(g+1) = m(g) + cm* sum( wi * (x(g+1) − m(g)))  ale dla cm = 1 ten i ten na górze są takie same 
        # xmean_new = xmean_old + cm * np.sum(weights * (x_sorted - m_old), axis=0)
        
        # wektor przesunięcia średniej
        y = (self.xmean - x_old) / self.sigma

        # z = y @ self.inv_sqrt
        
        # Aktualizacja ścieżki ewolucji dla sigmy
        # c^-1/2 = self.eigen_vectors*np.diag(self.eigen_values**0.5) * self.eigen_vectors.T
        self.ps = self.ps * (1-self.cs) + np.sqrt(self.cc * (2-self.cc)/self.mueff) * self.eigen_vectors @ np.diag(self.eigen_values**0.5) @ self.eigen_vectors.T @ y
 
        # Aktualizacja ścieżki ewolucji dla macierzy kowariancji
        self.pc = (1-self.cc) * self.pc + np.sqrt(self.cc * (2-self.cc)/self.mueff) * y
        
        #Aktualizacja macierzy kowariancji

        # suma ważona macierzy... 
        # macierz y*y.T i suma warzona po wagach to chyba po protsu y*y.T bo wagi sumują sie do 1
        # Rank-1 update (oparte na ścieżce ewolucji pc)
        rank_one_update = self.c1 * np.outer(self.pc, self.pc)

        # Rank-μ update (ważona suma macierzy y*y^T)
        weighted_sum = np.zeros_like(self.C)
        for k in range(len(self.weights)):
            y_k = (x[k] - x_old) / self.sigma  # Wektor y_k dla potomka k
            weighted_sum += self.weights[k] * np.outer(y_k, y_k)

        rank_mu_update = weighted_sum * self.cmu

        # zamiast 1 powinna być suma wag ale wiemy że sumują sie do 1
        self.C = (1-self.cc - self.cmu * 1) * self.C + self.c1 * rank_one_update + rank_mu_update

        self.C = (self.C + self.C.T)/2.0 # Upewniamy się, że macierz jest symetryczna
        
        # Aktualizacja rozmiaru kroku
        
        self.sigma = self.sigma * np.exp(self.cs * np.linalg.norm(self.ps)/np.sqrt(self.dimension) - 1)
        if(self.dimension==2):
            visualize_generation(x,self.fitvals,x_old,C_old,self.xmean,self.C,g)
        
        
    def terminate(self):
        """Zakończ algorytm"""
        if self.counteval <= 0:
            return False
        if self.counteval >= self.maxfevals:
            return True
        if self.condition_number > 1e13:
            return True
        if self.sigma * np.max(self.eigen_values)**0.5 < 1e-13:
            return True
        return False
    
def optimize(func, x0, sigma, maxfevals = 1000, popsize = None, weights = None):
    cma_es = CMA_ES(x0, sigma, maxfevals, popsize, weights)
    res = []
    cntr = 0
    while not cma_es.terminate():
        cntr+=1
        x = cma_es.sample()
        print(x)
        f_eval = func(x)
        print(f_eval)
        cma_es.update(x, f_eval,g=cntr)
        res.append(cma_es.best)
        if cntr % 100 == 0:
            print(f"Iteration {cntr:5d}: {res[-1][1]}")
    return res
def optimize_and_plot(f, sigma = 1, d = 10, popsize = None):
    x0 = np.repeat(100.0, d)
    res = optimize(f, x0, sigma, popsize = popsize)
    print(f"Best: {res[-1][0]}, value: {res[-1][1]}")
    y = np.array([nd for st, nd in res])
    fig = px.line(x = np.arange(y.shape[0]) + 1, y = y)
    fig.show()
    

#Ewaluacja CMA-ES

#optimize_and_plot(sphere_function, d = 2)
#optimize_and_plot(rastrigin_function, d=2)
#optimize_and_plot(ellipsoid_function, d=2)
#optimize_and_plot(cigar_function, d=2)

