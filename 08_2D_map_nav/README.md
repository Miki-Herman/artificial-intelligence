# Pathfinding Algorithms Visualization

Tato aplikace slouží k vizualizaci různých plánovacích algoritmů pro hledání cest v prostředí s překážkami.

## Popis

Aplikace využívá knihovnu Pygame k vizualizaci průběhu algoritmů pro hledání nejkratší cesty (pathfinding) v 2D mřížce. UFO se pohybuje v prostředí s překážkami a snaží se najít cestu k cíli pomocí jednoho ze tří implementovaných algoritmů:

- Dijkstrův algoritmus
- Algoritmus A*
- Greedy Best-First Search

## Funkce

- Interaktivní vizualizace pohybu UFO v prostředí
- Implementace tří různých plánovacích algoritmů
- Vizualizace expandovaných uzlů během hledání cesty
- Možnost nastavení startovní a cílové pozice
- Možnost přidávání překážek různých typů

## Požadavky

- Python 3.x
- pygame
- numpy
- collections (standardní knihovna)

## Instalace

```bash
pip install -r requirements.txt
```

## Použití

1. Ujistěte se, že máte v adresáři s programem následující obrázky:
    - tile.jpg
    - markedtile.jpg
    - house1.jpg, house2.jpg, house3.jpg
    - tree1.jpg, tree2.jpg
    - ufo.jpg
    - flag.jpg

2. Spusťte program:
```bash
python block_world.py
```

3. Program zobrazí UFO pohybující se po naplánované cestě k cíli.

## Úprava prostředí

V kódu můžete upravit následující parametry:

- Velikost prostředí (`WIDTH` a `HEIGHT`)
- Pozice překážek (`env.add_block(x, y)`)
- Startovní a cílovou pozici (`env.set_start(x, y)` a `env.set_goal(x, y)`)
- Použitý algoritmus pro plánování cesty (`env.path_planner(algorithm="dijkstra|a_star|greedy")`)
- Rychlost pohybu UFO (pomocí konstanty `FPS`)

## Implementované algoritmy

### Dijkstrův algoritmus

Klasický algoritmus pro hledání nejkratší cesty ve váženém grafu. V této implementaci má každý krok stejnou váhu (1), takže algoritmus hledá cestu s nejmenším počtem kroků.

### A* (A-star)

Rozšíření Dijkstrova algoritmu, které využívá heuristickou funkci k odhadu vzdálenosti k cíli. To umožňuje rychlejší nalezení cesty ve většině případů. Implementace používá Manhattanskou vzdálenost jako heuristiku.

### Greedy Best-First Search

Algoritmus, který vždy expanduje uzel, který se podle heuristiky zdá být nejblíže cíli. Na rozdíl od A* algoritmu nebere v úvahu již ujetou vzdálenost.

## Struktura kódu

- Třída `Env` - reprezentuje prostředí s překážkami
- Třída `Ufo` - reprezentuje agenta (UFO), který se pohybuje v prostředí
- Metody pro plánování cesty (`dijkstra`, `a_star`, `greedy_best_first`)
- Funkce pro vykreslování a obsluhu událostí
