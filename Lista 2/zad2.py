import numpy as np




def OX(parent1, parent2):
    start, end =np.sort(np.random.choice(len(parent1), 2, replace=False))
    length = len(parent1)
    child1 = [-1] * length  # Dziecko z brakującymi wartościami
    child2 = [-1] * length  # Dziecko z brakującymi wartościami
    
    # Kopiowanie segmentu z parent1 do dziecka
    child1[start:end] = parent1[start:end]
    child2[start:end] = parent2[start:end]
    
    # Uzupełnianie pozostałych wartości w dziecku na podstawie kolejności z parent2
    pos = end % length
    for gene in parent2:
        if gene not in child1:
            child1[pos] = gene
            pos = (pos + 1) % length
    
    pos = end % length
    for gene in parent1:
        if gene not in child2:
            child2[pos] = gene
            pos = (pos + 1) % length

    return np.array(child1),np.array(child2)


def CX(parent1, parent2):
    length = len(parent1)
    child1 = [-1] * length  # Początkowo dziecko bez wartości
    
    # Znalezienie cykli
    cycle = 0
    visited = [False] * length
    while -1 in child1:
        if cycle % 2 == 0:
            # Kopiowanie cyklu z parent1 do dziecka
            idx = child1.index(-1)
            while not visited[idx]:
                child1[idx] = parent1[idx]
                visited[idx] = True
                idx = np.where(parent1 == parent2[idx])[0][0]
        else:
            # Kopiowanie cyklu z parent2 do dziecka
            idx = child1.index(-1)
            while not visited[idx]:
                child1[idx] = parent2[idx]
                visited[idx] = True
                idx = np.where(parent2 == parent1[idx])[0][0]
        cycle += 1


    child2 = [-1] * length  # Początkowo dziecko bez wartości
    
    # Znalezienie cykli
    cycle = 0
    visited = [False] * length
    while -1 in child2:
        if cycle % 2 == 0:
            # Kopiowanie cyklu z parent1 do dziecka
            idx = child2.index(-1)
            while not visited[idx]:
                child2[idx] = parent2[idx]
                visited[idx] = True
                idx = np.where(parent2 == parent1[idx])[0][0]
        else:
            # Kopiowanie cyklu z parent2 do dziecka
            idx = child2.index(-1)
            while not visited[idx]:
                child2[idx] = parent1[idx]
                visited[idx] = True
                idx = np.where(parent1 == parent2[idx])[0][0]
        cycle += 1

    
    return np.array(child1), np.array(child2)


def PBX(parent1,parent2):
    def PBX_help(parent1, parent2):
        length = len(parent1)
        child1 = [-1] * length
    
        # Losowy wybór pozycji do przeniesienia
        positions = np.random.choice(range(length), size=length // 2, replace=False)
        for pos in positions:
            child1[pos] = parent1[pos]

        # Uzupełnienie pozostałych pozycji wartościami z parent2
        pos_parent2 = 0
        for i in range(length):
            if child1[i] == -1:
                while parent2[pos_parent2] in child1:
                    pos_parent2 += 1
                child1[i] = parent2[pos_parent2]
    
        return np.array(child1)
    return PBX_help(parent1,parent2),PBX_help(parent2,parent1)


def OBX(parent1, parent2):
    length = len(parent1)
    child = [-1] * length
    
    # Losowy wybór pozycji do przeniesienia
    positions = np.random.choice(range(length), size=length // 2, replace=False)
    for pos in positions:
        child[pos] = parent1[pos]
    
    # Uzupełnianie w kolejności pozostałymi wartościami z parent2
    pos_child = 0
    for val in parent2:
        if val not in child:
            while child[pos_child] != -1:
                pos_child += 1
            child[pos_child] = val
            
    return np.array(child)


def PPX(parent1,parent2):
    def PPX_help(parent1, parent2):
        length = len(parent1)
        child = []
        
        while len(child) < length:
            if np.random.rand() < 0.5:
                for gene in parent1:
                    if gene not in child:
                        child.append(gene)
                        break
            else:
                for gene in parent2:
                    if gene not in child:
                        child.append(gene)
                        break
        
        return np.array(child)
    return PPX_help(parent1,parent2) , PPX_help(parent2,parent1)


def LCSX(parent1, parent2):
    length = len(parent1)
    
    # Znajdź największą wspólną sekwencję (najprostsza wersja)
    lcs = [gene for gene in parent1 if gene in parent2]
    
    # Uzupełnianie potomka
    child = [-1] * length
    for i, gene in enumerate(parent1):
        if gene in lcs:
            child[i] = gene
    
    pos_parent2 = 0
    for i in range(length):
        if child[i] == -1:
            while parent2[pos_parent2] in child:
                pos_parent2 += 1
            child[i] = parent2[pos_parent2]
    
    return np.array(child)


def LOX(parent1, parent2):
    start, end =np.sort(np.random.choice(len(parent1), 2, replace=False))
    length = len(parent1)
    child = [-1] * length
    
    # Kopiowanie segmentu z parent1
    child[start:end] = parent1[start:end]
    
    # Uzupełnianie potomka elementami z parent2 w kolejności liniowej
    pos = end % length
    for gene in parent2:
        if gene not in child:
            child[pos] = gene
            pos = (pos + 1) % length
    
    return np.array(child)


def NR(permutation):
    return np.random.permutation(len(permutation))

def NR_K(permutation):
    k = np.random.choice(np.arange(len(permutation)-2)) + 2
    indices = np.random.choice(len(permutation),k,replace=False)
    values_to_swap = permutation[indices].copy()
    np.random.shuffle(values_to_swap)
    permutation[indices] = values_to_swap
    return permutation