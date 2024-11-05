import numpy as np
import itertools as it


# Funkcja generująca sąsiadów poprzez zamianę do K pozycji w permutacji

def generate_neighbors(permutation, K):
    results = set()  # Używamy zbioru, aby uniknąć duplikatów
    n = len(permutation)
    
    def swap_and_collect(current, swaps_left, depth):
        # Dodaj obecną permutację do wyników
        results.add(tuple(current))
        
        # Jeśli nie mamy już zamian do wykonania, zwróć
        if swaps_left == 0:
            return
        
        # Przeprowadź zamiany
        for i in range(n):
            for j in range(i + 1, n):  # Zamiana tylko dla i < j
                # Zamień dwa elementy
                current[i], current[j] = current[j], current[i]
                
                # Rekursywnie zamień dalej
                swap_and_collect(current, swaps_left - 1, depth + 1)
                
                # Przywróć oryginalny stan
                current[i], current[j] = current[j], current[i]

    swap_and_collect(permutation.copy(), K, 0)
    return np.array([list(perm) for perm in results])  # Konwertuj krotki z powrotem na listy

# Mutacja z przeszukiwaniem lokalnym (jednokrotna zamiana na najlepszą z sąsiadów)
def local_search_mutation(tsp_objective_function,permutation, K):
    neighbors = generate_neighbors(permutation, K)
    best_neighbor = permutation
    best_value = tsp_objective_function(permutation)
    
    for neighbor in neighbors:
        neighbor_value = tsp_objective_function(neighbor)
        if neighbor_value < best_value:  # Zakładamy minimalizację funkcji celu
            best_neighbor = neighbor
            best_value = neighbor_value
            
    return best_neighbor

# Iterowane przeszukiwanie lokalne
def iterated_local_search_mutation(tsp_objective_function,permutation, K):
    improved = True
    current_permutation = permutation
    current_value = tsp_objective_function(current_permutation)
    number_of_iters=0
    while improved:
        improved = False
        neighbors = generate_neighbors(current_permutation, K)
        neighbors_count = len(neighbors)
        #print(f'do przeszukania {neighbors_count}')
        #print(f'curr_perm: {permutation}')
        #print(f'curr_max_val {current_value}')
        for neighbor in neighbors:
            neighbor_value = tsp_objective_function(neighbor)
            if neighbor_value < current_value:  # Minimalizacja funkcji celu
                current_permutation = neighbor
                current_value = neighbor_value
                improved = True
        number_of_iters+=1
        #print(f'number_of_iters: {number_of_iters}')


    return current_permutation
