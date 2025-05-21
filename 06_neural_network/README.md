# Boom Master s neuronovou sítí a evolucí

Tento projekt implementuje evoluční algoritmus a neuronové sítě pro kontrolu agentů ve hře "Boom Master". Agenti se učí navigovat prostředím s minami a dosáhnout cílové vlajky.

## Popis implementace

Implementace rozšiřuje původní hru "Boom Master" o následující komponenty:

### 1. Senzorické funkce

Funkce `get_agent_sensors` implementuje senzorické schopnosti agentů, což bylo požadováno v první části zadání. Poskytuje agentům následující vstupy:

- 8 senzorů směru k nejbližší mině (S, SV, V, JV, J, JZ, Z, SZ)
- Vzdálenost k vlajce
- Úhel k vlajce

Tyto senzory poskytují agentovi všechny potřebné informace pro efektivní navigaci prostředím.

```python
def get_agent_sensors(agent, mines):
    """
    Získá senzorické vstupy pro agenta. Vrací seznam vstupů:
    - Vzdálenost k nejbližší mině v 8 směrech (normalizovaná)
    - Vzdálenost k vlajce (normalizovaná)
    - Směrový úhel k vlajce (normalizovaný)
    """
```

### 2. Neuronová síť

Funkce `nn_function(inp, wei)` implementuje dopřednou neuronovou síť, což odpovídá druhé části zadání. Struktura sítě:

- Vstupní vrstva (10 neuronů pro senzorické vstupy)
- Skrytá vrstva (8 neuronů)
- Výstupní vrstva (4 neurony pro rozhodnutí o pohybu)

Funkce přijímá vstupní vektor a váhové matice, provádí maticové násobení s aktivačními funkcemi a vrátí výstup.

```python
def nn_function(inp, weights):
    """
    Zpracování vstupů skrz neuronovou síť s danými vahami.
    
    Args:
        inp: Vstupní vektor (senzorické vstupy)
        weights: Seznam váhových matic [w1, w2], kde:
            - w1: Váhy ze vstupní do skryté vrstvy
            - w2: Váhy ze skryté do výstupní vrstvy
            
    Returns:
        Výstupní vektor (rozhodnutí o pohybu)
    """
```

### 3. Navigační funkce

Funkce `nn_navigate_me(agent, inp, mines)` implementuje navigaci agenta na základě výstupů neuronové sítě, což bylo požadováno ve třetí části zadání:

- Získá výstupy neuronové sítě na základě senzorických vstupů
- Interpretuje výstupy jako rozhodnutí o pohybu (nahoru, dolů, vlevo, vpravo)
- Aktualizuje pozici agenta podle těchto rozhodnutí
- Sleduje, zda se agent přibližuje k vlajce nebo se od ní vzdaluje

```python
def nn_navigate_me(agent, inp, mines):
    """
    Naviguje agenta na základě výstupů neuronové sítě
    
    Args:
        agent: Agent, který se má navigovat
        inp: Senzorické vstupy
        
    Returns:
        None (přímo aktualizuje pozici agenta)
    """
```

### 4. Výpočet fitness

Funkce `handle_mes_fitnesses(agents, mines, steps_elapsed)` implementuje výpočet fitness pro každého agenta, což odpovídá čtvrté části zadání:

#### Fitness funkce:

Fitness každého agenta se skládá z několika komponent:

1. **Základní fitness** - počet kroků, které agent přežil
2. **Bonus za dosažení vlajky** - velký bonus (10000 bodů), pokud agent dosáhl vlajky
3. **Bonus za rychlost** - dodatečný bonus za rychlé dosažení vlajky (5000 - počet_kroků)
4. **Bonus za přibližování k vlajce** - 20 bodů za každý krok směrem k vlajce (50 bodů, pokud nakonec dosáhl vlajky)
5. **Penalizace za vzdalování od vlajky** - snížení počítadla kroků směrem k vlajce o 0.5 za každý krok od vlajky
6. **Gradient vzdálenosti** - pokud agent nedosáhl vlajky, získá body úměrné k tomu, jak blízko se k ní dostal

Tato fitness funkce silně odměňuje agenty, kteří dosáhnou vlajky, a poskytuje gradient informací pro ty, kteří ji nedosáhnou, což pomáhá evolučnímu algoritmu konvergovat k efektivnímu řešení.

```python
def handle_mes_fitnesses(agents, mines, steps_elapsed):
    """
    Vypočítá fitness pro každého agenta.
    
    Args:
        agents: Seznam agentů
        mines: Seznam min
        steps_elapsed: Počet uplynulých kroků v simulaci
        
    Returns:
        Seznam aktualizovaných agentů s vypočítanou fitness
    """
```

### 5. Evoluční algoritmus

Parametry evolučního algoritmu byly nastaveny podle páté části zadání:

- Velikost populace: 30 agentů
- Pravděpodobnost mutace: 0.2 (20% šance na mutaci každé váhy)
- Síla mutace: 0.3 (maximální hodnota změny váhy)
- Velikost elity: 5 (nejlepších 5 agentů je zachováno beze změny)
- Maximální počet generací: 100
- Struktura neuronové sítě: 10 vstupů, 8 skrytých neuronů, 4 výstupy

#### Selekce

Pro výběr rodičů nové generace je použita turnajová selekce, která vybírá lepší jedince s vyšší pravděpodobností, ale dává šanci i slabším:

```python
def selection(population):
    """Turnajová selekce"""
    selected = []
    
    # Zachování elitních jedinců
    sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
    selected.extend(copy.deepcopy(sorted_pop[:ELITE_SIZE]))
    
    # Turnajová selekce pro zbytek
    while len(selected) < POPULATION_SIZE:
        tournament_size = 3
        tournament = random.sample(population, tournament_size)
        winner = max(tournament, key=lambda x: x.fitness)
        selected.append(copy.deepcopy(winner))
    
    return selected
```

#### Křížení

Pro křížení je implementována metoda jednobodového křížení pro každou váhovou matici:

```python
def crossover(parent1, parent2):
    """Provede křížení mezi dvěma rodiči"""
    child = Agent()
    
    # Křížení pro weights1
    rows, cols = parent1.weights1.shape
    crossover_point_row = random.randint(0, rows-1)
    crossover_point_col = random.randint(0, cols-1)
    
    for i in range(rows):
        for j in range(cols):
            if i < crossover_point_row or (i == crossover_point_row and j <= crossover_point_col):
                child.weights1[i][j] = parent1.weights1[i][j]
            else:
                child.weights1[i][j] = parent2.weights1[i][j]
    
    # Podobně pro weights2
    # ...
    
    return child
```

#### Mutace

Mutace mění náhodné váhy s pravděpodobností MUTATION_RATE a silou MUTATION_STRENGTH:

```python
def mutation(agent):
    """Zmutuje genom agenta"""
    # Mutace weights1
    rows, cols = agent.weights1.shape
    for i in range(rows):
        for j in range(cols):
            if random.random() < MUTATION_RATE:
                agent.weights1[i][j] += random.uniform(-MUTATION_STRENGTH, MUTATION_STRENGTH)
    
    # Podobně pro weights2
    # ...
    
    return agent
```

### 6. Integrace do hry Boom

Implementace byla plně integrována do hry Boom podle šesté části zadání, zahrnujíc:

- Třídu `Agent`, která rozšiřuje původního hráče o neuronové sítě
- Evoluční funkce (selekce, křížení, mutace)
- Simulaci pro každou generaci
- Sledování a ukládání nejlepšího genomu
- Vizualizaci evolučního procesu

## Závěr

Tento projekt ukazuje implementaci evolučního algoritmu a neuronových sítí pro řízení agentů ve hře. Agenti začínají s náhodnými chováními a postupně se vyvíjejí přes generace, aby se lépe naučili navigovat minovým polem a dosáhnout vlajky.

Klíčovou inovací je přidání fitness komponenty, která odměňuje agenty za přibližování se k vlajce a penalizuje je za vzdalování se. Toto významně zlepšuje učení, protože poskytuje mnohem jasnější gradient pro evoluční algoritmus.