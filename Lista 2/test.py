import numpy as np

from zad2 import OX,CX,PBX,OBX,PPX,LCSX,LOX,NR_K,NR
from zad3 import iterated_local_search_mutation,generate_neighbors

def PMX(ind1, ind2):
    # Długość genotypu
    genome_length = len(ind1)
    
    print

    # Wybór dwóch punktów krzyżowania
    start, end = np.sort(np.random.choice(genome_length, 2, replace=False))
    
    # Tworzenie potomków na podstawie rodziców
    child1, child2 = ind1.copy(), ind2.copy()
    
    # Przenoszenie segmentów
    child1[start:end] = ind2[start:end]
    child2[start:end] = ind1[start:end]
    
    def resolve_conflicts(child, parent, start, end):
        for i in range(start, end):
            # Check for duplicates outside the crossover segment
            if parent[i] not in child[start:end]:
                current_val = parent[i]
                index = i
                
                # Map values until there is no conflict
                while current_val in child:
                    index = np.where(parent == current_val)[0][0]
                    current_val = child[index]
                
                # Place the mapped value in the child's position
                child[index] = parent[i]
    
    resolve_conflicts(child1, ind1, start, end)
    resolve_conflicts(child2, ind2, start, end)
    
    return child1, child2

def transpose_mutation(p):
    i , j = np.sort(np.random.choice(len(p), 2, replace=False))
    p[i] = p[i] + p[j]
    p[j] = p[i] - p[j]
    p[i] = p[i] - p[j]
    return p

a=np.arange(10)
b=np.array([0,5,4,3,2,1,6,7,8,9])





