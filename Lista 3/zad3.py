import numpy as np
import matplotlib.pyplot as plt


def ES_both_gens(mi,lam , size_of_one_guy , goal_fun , bounds = (-np.inf,np.inf) , if_recombine = False , recombine_fun = None , iterations=1000 , both=True ):
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
        if(bounds[1]==np.inf and bounds[0]==-np.inf):
            parent_population = np.random.uniform(-1e10,1e10,(mi,size_of_one_guy)) 
        elif(bounds[1]==np.inf and bounds[0]!=-np.inf):
            parent_population = np.random.uniform(bounds[0],1e10,(mi,size_of_one_guy))
        elif(bounds[1]!=np.inf and bounds[0]==-np.inf):
            parent_population = np.random.uniform(-1e10,bounds[1],(mi,size_of_one_guy))
        if(bounds[1]!=np.inf and bounds[0]!=-np.inf):
            parent_population = np.random.uniform(bounds[0],bounds[1],(mi,size_of_one_guy))

        init_mean=np.mean(np.array([goal_fun(parent_population[i]) for i in range(len(parent_population))]))

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
        fitness = np.array([goal_fun(population[i]) for i in range(len(population))])
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
