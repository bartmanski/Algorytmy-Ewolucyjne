import numpy as np
import scipy
import scipy.linalg
from scipy.stats import chi2
import plotly.graph_objs as go
import plotly.express as px


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
    
    def update(self, x, fitvals):
        """Zaktualizuj wartości uzyskanych parametrów"""
        self.counteval += fitvals.shape[0] # Zwiększamy licznik wykonań
        N = self.xmean.shape[0]
        x_old = self.xmean.copy()
        
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
        self.ps = self.ps * (1-self.cs) + np.sqrt(self.cc * (2-self.cc)/self.mueff) * self.eigen_vectors @ np.diag(1/np.sqrt(self.eigen_values)) @ self.eigen_vectors.T @ y
 
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
            y_k = np.real((x[k] - x_old) / self.sigma)  # Wektor y_k dla potomka k
            weighted_sum += self.weights[k] * np.outer(y_k, y_k)

        rank_mu_update = weighted_sum * self.cmu

        # zamiast 1 powinna być suma wag ale wiemy że sumują sie do 1
        self.C = (1-self.cc - self.cmu * 1) * self.C + self.c1 * rank_one_update + rank_mu_update

        self.C = (self.C + self.C.T)/2.0 # Upewniamy się, że macierz jest symetryczna
        self.C = np.real(self.C)
        
        # Aktualizacja rozmiaru kroku
        
        self.sigma = self.sigma * np.exp(self.cs * np.linalg.norm(self.ps)/np.sqrt(self.dimension) - 1)
        self.sigma = self.sigma * np.sqrt(self.dimension)
        self.sigma = np.real(self.sigma) 
  
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

def ES_both_gens(mi,lam , size_of_one_guy , goal_fun ,if_init_pop=False,init_population=None ,bounds = (-np.inf,np.inf) , if_recombine = False , recombine_fun = None , iterations=1000 , both=True ):
    # mi - jak dużo osbników w generacji początkowej / rodzicach
    # lam - jak dużo dzieci powstaje
    # size_of_one_guy  - jak duży jest osobnik ( w alg ewo osobnik z populacji jest jakimś wektorem z R^n i wielkość osobnika to jest dokładnie to n) [jak dużo liczb potrzeba żeby zdefiniować osobnika]
    # goal_function - funkcja celu, funkcja na podstawie której sprawdzamy który osobnik jest lepiej przystodsowany (który jest blizej opta)
    # bounds - granice dziedzin naszych funkcji , funkcja celu może mieć ograniczoną dziedzine przez co chcemy aby nasze osobniki sie w niej mieściły
    # if_recombine - jest szansa ze chcemy żeby potomstwo było tworzone z 2 rodziców i było jakąś ich mieszanką wtedy zmieniamy to na true i ustalamy jak to robimy w recombine_fun
    # recombine_fun - funkcja przyjmująca dwójkę rodziców i zwracająca i jakąś kobinacje jako dziecko

    def initialize():
        # jak bardzo bedziemy zaburzac dane
        if(bounds[1]==np.inf):
            max_mutation_tweak= 10000
        else:
            max_mutation_tweak= bounds[1]/10000

        # generacja populacji startowej
        if(if_init_pop):
            parent_population=init_population
        else:
            if(bounds[1]==np.inf and bounds[0]==-np.inf):
                parent_population = np.random.uniform(-1e10,1e10,(mi,size_of_one_guy)) 
            elif(bounds[1]==np.inf and bounds[0]!=-np.inf):
                parent_population = np.random.uniform(bounds[0],1e10,(mi,size_of_one_guy))
            elif(bounds[1]!=np.inf and bounds[0]==-np.inf):
                parent_population = np.random.uniform(-1e10,bounds[1],(mi,size_of_one_guy))
            if(bounds[1]!=np.inf and bounds[0]!=-np.inf):
                parent_population = np.random.uniform(bounds[0],bounds[1],(mi,size_of_one_guy))

        init_mean=np.mean(goal_fun(parent_population))

        return max_mutation_tweak,parent_population,init_mean
    
    def gen_kids(parents,max_mutation_tweak):

        def gen_kid(parent):
            return parent + np.random.normal(0,max_mutation_tweak,size_of_one_guy)
        
        def gen_kid_corss(parent1,parent2):
            
            child= recombine_fun(parent1,parent2)
            return child + np.random.normal(0,max_mutation_tweak,size_of_one_guy)

        new_population=[]
        if(if_recombine):
            for i in range(lam):
                parents = parents[np.random.choice(mi,size=2,replace=False)]
                new_population.append(gen_kid_corss(parents[0],parents[1]))
        else:
            for i in range(lam):
                parent=np.random.choice(mi,1)[0]
                new_population.append(gen_kid(parent))
    
        return np.clip(new_population,bounds[0],bounds[1])
    
    def cut_population_to_mi(parents,kids):
        if(both):
            population=np.vstack((parents,kids))
        else:
            population=kids
        fitness = goal_fun(population)
        Indeces_sorted = np.argsort(fitness)
        sorted = population[Indeces_sorted]
        mean_score = np.mean(fitness)
        min = fitness[Indeces_sorted[0]]
        return sorted[:mi],mean_score,min

    def adapt_mutation(max_mutation_tweak , prev_mean , curr_mean , iter):
        increase = 3 * (1 - iter/iterations)
        decrease = 0.9999 * (1 - iter/iterations)
        if ( curr_mean > prev_mean):
            # Mutacje dają efekty 
            return max_mutation_tweak * increase
        # jeśli nie to jeszcze nie doszło do zbiezności więc jeśli mutacje są dobre (curr>prew) to chcemy żeby były większe a jeśli złe to żeby się zmniejszały
        return max_mutation_tweak*decrease
    
    max_mutation_tweak,parent_population,prev_mean=initialize()

    means=np.zeros(iterations)
    mins=np.zeros(iterations)

    for i in range(iterations):
        
        kids=gen_kids(parent_population,max_mutation_tweak)
        parent_population,new_mean,min=cut_population_to_mi(kids,parent_population)
        mins[i] = min
        means[i] = new_mean
        max_mutation_tweak=adapt_mutation(max_mutation_tweak,prev_mean,new_mean,i)
        prev_mean=new_mean

    return parent_population,mins,means

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
        cma_es.update(x, f_eval)
        res.append(cma_es.best)
        if cntr % 100 == 0:
            print(f"Iteration {cntr:5d}: {res[-1][1]}")
    return res
def optimize_plot_and_compare(f, sigma = 1, d = 10, popsize = None):
    x0 = np.repeat(100.0, d)
    res = optimize(f, x0, sigma, popsize = popsize)
    print(f"Best: {res[-1][0]}, value: {res[-1][1]}")
    y = np.array([nd for st, nd in res])
    fig = px.line(x = np.arange(y.shape[0]) + 1, y = y)
    fig.show()


list_of_benchmarks=[sphere_function,ellipsoid_function,rastrigin_function,schwefel_function,griewank_function,cigar_function,discus_function,cigar_discus_function,parab_ridge_function,two_axes_function]
list_of_benchmarks_names=['sphere_function','ellipsoid_function','rastrigin_function','schwefel_function','griewank_function','cigar_function','discus_function','cigar_discus_function','parab_ridge_function','two_axes_function']
list_of_dims=[5,10,20,50]
results={}
'''
for index,fun in enumerate(list_of_benchmarks):
    name=list_of_benchmarks_names[index]
    fun_result_sigma1=[]
    fun_result_sigma5=[]
    fun_result_sigma10=[]
    
    for dims in list_of_dims:
        x0 = np.repeat(100.0, dims)
        fun_result_sigma1.append(optimize(fun,x0,sigma=1))
        fun_result_sigma5.append(optimize(fun,x0,sigma=5))
        fun_result_sigma10.append(optimize(fun,x0,sigma=10))
   
    results[name]=[fun_result_sigma1,fun_result_sigma5,fun_result_sigma10]
''' 
#np.save('results_all_funs_dict',results)
#results=np.load('results_all_funs_dict.npy',allow_pickle=True).item() 
#print(results)

results={}
for index,fun in enumerate(list_of_benchmarks):
    name=list_of_benchmarks_names[index]
    es_result=[]
    for dims in list_of_dims:
        x0 = np.tile(100.0, (100, dims)) 
        es_result.append((ES_both_gens(100,200,dims,fun,True,x0)))
    results[name]=es_result
np.save('results_ES',results)