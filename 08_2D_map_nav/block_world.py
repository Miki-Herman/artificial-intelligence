# -*- coding: utf-8 -*-
import pygame
import random

import numpy as np
from collections import deque

BLOCKTYPES = 5

# třída reprezentující prostředí
class Env:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.arr = np.zeros((height, width), dtype=int)
        self.startx = 0
        self.starty = 0
        self.goalx = width-1
        self.goaly = height-1
        
    def is_valid_xy(self, x, y):      
        if x >= 0 and x < self.width and y >= 0 and y < self.height and self.arr[y, x] == 0:
            return True
        return False 
        
    def set_start(self, x, y):
        if self.is_valid_xy(x, y):
            self.startx = x
            self.starty = y
            
    def set_goal(self, x, y):
        if self.is_valid_xy(x, y):
            self.goalx = x
            self.goaly = y
               
        
    def is_empty(self, x, y):
        if self.arr[y, x] == 0:
            return True
        return False
    
        
    def add_block(self, x, y):
        if self.arr[y, x] == 0:
            r = random.randint(1, BLOCKTYPES)
            self.arr[y, x] = r
                
    def get_neighbors(self, x, y):
        l = []
        if x-1 >= 0 and self.arr[y, x-1] == 0:
            l.append((x-1, y))
        
        if x+1 < self.width and self.arr[y, x+1] == 0:
            l.append((x+1, y))
            
        if y-1 >= 0 and self.arr[y-1, x] == 0:
            l.append((x, y-1))
        
        if y+1 < self.height and self.arr[y+1, x] == 0:
            l.append((x, y+1))
        
        return l
        
     
    def get_tile_type(self, x, y):
        return self.arr[y, x]

    # --------- Pomocné funkce pro plánovací funkce
    def _heuristic(self, x1, y1, x2, y2):
        """
        Výpočet Manhattan vzdálenosti mezi dvěma body.

        Args:
            x1 (int): X-souřadnice prvního bodu.
            y1 (int): Y-souřadnice prvního bodu.
            x2 (int): X-souřadnice druhého bodu.
            y2 (int): Y-souřadnice druhého bodu.

        Returns:
            int: Manhattan vzdálenost mezi body.
        """
        return abs(x1 - x2) + abs(y1 - y2)


    def _reconstruct_path(self, came_from, goal_x, goal_y):
        """
        Rekonstruuje cestu od cíle k počátku a vrátí ji jako frontu.

        Args:
            came_from (list): 2D pole předchůdců bodů.
            goal_x (int): X-souřadnice cílového bodu.
            goal_y (int): Y-souřadnice cílového bodu.

        Returns:
            deque: Fronta s cestou od počátku k cíli.
        """
        path = deque()
        x, y = goal_x, goal_y

        # Přidání cílového bodu do cesty
        path.appendleft((x, y))

        # Procházení předchůdců až k počátečnímu bodu
        while came_from[y][x] != (-1, -1):
            x, y = came_from[y][x]
            path.appendleft((x, y))

        return path

    
    # vrací dvojici 1. frontu dvojic ze startu do cíle, 2. seznam dlaždic
    # k zobrazení - hodí se např. pro zvýraznění cesty, nebo expandovaných uzlů
    # start a cíl se nastaví pomocí set_start a set_goal
    # <------    ZDE vlastní metoda
    def dijkstra(self, start, goal):
        # Inicializace 2D pole nákladů s nekonečnem
        width = self.width
        height = self.height

        start_x = start[0]
        start_y = start[1]

        goal_x = goal[0]
        goal_y = goal[1]

        cost = [[float('inf') for _ in range(width)] for _ in range(height)]
        cost[start_y][start_x] = 0

        # Inicializace 2D pole předchůdců
        came_from = [[(-1, -1) for _ in range(width)] for _ in range(height)]

        # Seznam expandovaných uzlů pro vizualizaci
        expanded = []

        # Prioritní fronta jako seznam dvojic (náklady, (x, y))
        frontier = [(0, (start_x, start_y))]

        # Inicializace pole pro sledování navštívených uzlů
        visited = [[False for _ in range(width)] for _ in range(height)]

        # Hlavní smyčka algoritmu
        while len(frontier) > 0:
            # Výběr uzlu s nejnižšími náklady
            current_cost, (current_x, current_y) = frontier[0]
            frontier.pop(0)

            # Přeskočení již zpracovaných uzlů
            if visited[current_y][current_x]:
                continue

            # Označení uzlu jako navštíveného
            visited[current_y][current_x] = True

            # Přidání do seznamu expandovaných uzlů
            if (current_x != start_x or current_y != start_y) and (current_x != goal_x or current_y != goal_y):
                expanded.append((current_x, current_y))

            # Ukončení při dosažení cíle
            if current_x == goal_x and current_y == goal_y:
                break

            # Získání seznamu sousedů
            neighbors = self.get_neighbors(current_x, current_y)

            # Expanze sousedů
            for (next_x, next_y) in neighbors:
                if not visited[next_y][next_x]:
                    # Náklad přechodu = 1
                    new_cost = current_cost + 1

                    # Pokud byla nalezena levnější cesta
                    if new_cost < cost[next_y][next_x]:
                        cost[next_y][next_x] = new_cost
                        came_from[next_y][next_x] = (current_x, current_y)

                        # Vyhledání vhodné pozice ve frontě
                        i = 0
                        while i < len(frontier) and frontier[i][0] < new_cost:
                            i += 1

                        # Vložení na správné místo
                        frontier.insert(i, (new_cost, (next_x, next_y)))

        # Rekonstrukce cesty
        path = self._reconstruct_path(came_from, goal_x, goal_y)

        return path, expanded

    def a_star(self, start, goal):
        # Inicializace proměnných
        width = self.width
        height = self.height

        start_x = start[0]
        start_y = start[1]
        goal_x = goal[0]
        goal_y = goal[1]

        # Inicializace 2D pole g-skóre (nákladů ze startu) s nekonečnem
        g_score = [[float('inf') for _ in range(width)] for _ in range(height)]
        g_score[start_y][start_x] = 0

        # Inicializace 2D pole f-skóre (celkových nákladů) s nekonečnem
        f_score = [[float('inf') for _ in range(width)] for _ in range(height)]
        f_score[start_y][start_x] = self._heuristic(start_x, start_y, goal_x, goal_y)

        # Inicializace 2D pole předchůdců
        came_from = [[(-1, -1) for _ in range(width)] for _ in range(height)]

        # Seznam expandovaných uzlů pro vizualizaci
        expanded = []

        # Prioritní fronta jako seznam trojic (f-skóre, g-skóre, (x, y))
        # g-skóre je důležité pro případy, kdy f-skóre je stejné
        frontier = [(f_score[start_y][start_x], 0, (start_x, start_y))]

        # Inicializace pole pro sledování navštívených uzlů
        visited = [[False for _ in range(width)] for _ in range(height)]

        # Hlavní smyčka algoritmu
        while len(frontier) > 0:
            # Výběr uzlu s nejnižším f-skóre
            _, _, (current_x, current_y) = frontier[0]
            frontier.pop(0)

            # Přeskočení již zpracovaných uzlů
            if visited[current_y][current_x]:
                continue

            # Označení uzlu jako navštíveného
            visited[current_y][current_x] = True

            # Přidání do seznamu expandovaných uzlů
            if (current_x != start_x or current_y != start_y) and (current_x != goal_x or current_y != goal_y):
                expanded.append((current_x, current_y))

            # Ukončení při dosažení cíle
            if current_x == goal_x and current_y == goal_y:
                break

            # Získání seznamu sousedů
            neighbors = self.get_neighbors(current_x, current_y)

            # Expanze sousedů
            for (next_x, next_y) in neighbors:
                if not visited[next_y][next_x]:
                    # Výpočet g-skóre pro souseda (náklad přechodu = 1)
                    tentative_g_score = g_score[current_y][current_x] + 1

                    # Pokud byla nalezena levnější cesta
                    if tentative_g_score < g_score[next_y][next_x]:
                        # Aktualizace předchůdce
                        came_from[next_y][next_x] = (current_x, current_y)

                        # Aktualizace skóre
                        g_score[next_y][next_x] = tentative_g_score
                        f_score[next_y][next_x] = tentative_g_score + self._heuristic(next_x, next_y, goal_x, goal_y)

                        # Kontrola, zda uzel již není ve frontě
                        already_in_frontier = False
                        for i in range(len(frontier)):
                            if frontier[i][2] == (next_x, next_y):
                                # Aktualizace f-skóre ve frontě
                                frontier[i] = (f_score[next_y][next_x], g_score[next_y][next_x], (next_x, next_y))
                                already_in_frontier = True
                                break

                        # Přidání do fronty, pokud tam ještě není
                        if not already_in_frontier:
                            frontier.append((f_score[next_y][next_x], g_score[next_y][next_x], (next_x, next_y)))

                # Seřazení fronty podle f-skóre (primární klíč) a g-skóre (sekundární klíč)
                frontier.sort()

        # Rekonstrukce cesty
        path = self._reconstruct_path(came_from, goal_x, goal_y)

        return path, expanded

    def greedy_best_first(self, start, goal):
        # Inicializace proměnných
        width = self.width
        height = self.height

        start_x = start[0]
        start_y = start[1]
        goal_x = goal[0]
        goal_y = goal[1]

        # Inicializace 2D pole předchůdců
        came_from = [[(-1, -1) for _ in range(width)] for _ in range(height)]

        # Seznam expandovaných uzlů pro vizualizaci
        expanded = []

        # Prioritní fronta jako seznam dvojic (heuristika, (x, y))
        h_start = self._heuristic(start_x, start_y, goal_x, goal_y)
        frontier = [(h_start, (start_x, start_y))]

        # Inicializace pole pro sledování navštívených uzlů
        visited = [[False for _ in range(width)] for _ in range(height)]

        # Hlavní smyčka algoritmu
        while len(frontier) > 0:
            # Výběr uzlu s nejnižší heuristikou
            _, (current_x, current_y) = frontier[0]
            frontier.pop(0)

            # Přeskočení již zpracovaných uzlů
            if visited[current_y][current_x]:
                continue

            # Označení uzlu jako navštíveného
            visited[current_y][current_x] = True

            # Přidání do seznamu expandovaných uzlů
            if (current_x != start_x or current_y != start_y) and (current_x != goal_x or current_y != goal_y):
                expanded.append((current_x, current_y))

            # Ukončení při dosažení cíle
            if current_x == goal_x and current_y == goal_y:
                break

            # Získání seznamu sousedů
            neighbors = self.get_neighbors(current_x, current_y)

            # Expanze sousedů
            for (next_x, next_y) in neighbors:
                if not visited[next_y][next_x]:
                    # Výpočet heuristické hodnoty
                    h = self._heuristic(next_x, next_y, goal_x, goal_y)

                    # Zaznamenání předchůdce
                    came_from[next_y][next_x] = (current_x, current_y)

                    # Vyhledání vhodné pozice ve frontě
                    i = 0
                    while i < len(frontier) and frontier[i][0] < h:
                        i += 1

                    # Vložení na správné místo
                    frontier.insert(i, (h, (next_x, next_y)))

        # Rekonstrukce cesty
        path = self._reconstruct_path(came_from, goal_x, goal_y)

        return path, expanded


    def path_planner(self, algorithm="dijkstra"):

        start = (self.startx, self.starty)
        goal = (self.goalx, self.goaly)

        # Výběr algoritmu pro plánování cesty
        if algorithm == "dijkstra":
            return self.dijkstra(start, goal)
        elif algorithm == "greedy":
            return self.greedy_best_first(start, goal)
        else:
            return self.a_star(start, goal)
    
       
        
# třída reprezentující ufo        
class Ufo:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.path = deque()
        self.tiles = []
    
   
    # přemístí ufo na danou pozici - nejprve je dobré zkontrolovat u prostředí, 
    # zda je pozice validní
    def move(self, x, y):
        self.x = x
        self.y = y
    
    # reaktivní navigace <------------------------ !!!!!!!!!!!! ZDE DOPLNIT
    def reactive_go(self, env):
        r = random.random()
        
        dx = 0
        dy = 0
        
        if r > 0.5: 
            r = random.random()
            if r < 0.5:
                dx = -1
            else:
                dx = 1
            
        else:
            r = random.random()
            if r < 0.5:
                dy = -1
            else:
                dy = 1
        
        return (self.x + dx, self.y + dy)
        
    
    # nastaví cestu k vykonání 
    def set_path(self, p, t=[]):
        self.path = p
        self.tiles = t
   
    
    # vykoná naplánovanou cestu, v každém okamžiku na vyzvání vydá další
    # way point 
    def execute_path(self):
        if self.path:
            return self.path.popleft()
        return (-1, -1)

# definice prostředí -----------------------------------

TILESIZE = 50



#<------    definice prostředí a překážek !!!!!!

WIDTH = 12
HEIGHT = 9

env = Env(WIDTH, HEIGHT)

env.add_block(1, 1)
env.add_block(2, 2)
env.add_block(3, 3)
env.add_block(4, 4)
env.add_block(5, 5)
env.add_block(6, 6)
env.add_block(7, 7)
env.add_block(8, 8)
env.add_block(0, 8)

env.add_block(11, 1)
env.add_block(11, 6)
env.add_block(1, 3)
env.add_block(2, 4)
env.add_block(4, 5)
env.add_block(2, 6)
env.add_block(3, 7)
env.add_block(4, 8)
env.add_block(0, 8)


env.add_block(1, 8)
env.add_block(2, 8)
env.add_block(3, 5)
env.add_block(4, 8)
env.add_block(5, 6)
env.add_block(6, 4)
env.add_block(7, 2)
env.add_block(8, 1)


# pozice ufo <--------------------------
ufo = Ufo(env.startx, env.starty)

WIN = pygame.display.set_mode((env.width * TILESIZE, env.height * TILESIZE))

pygame.display.set_caption("Block world")

pygame.font.init()

WHITE = (255, 255, 255)


FPS = 2


# pond, tree, house, car

BOOM_FONT = pygame.font.SysFont("comicsans", 100)   
LEVEL_FONT = pygame.font.SysFont("comicsans", 20)   


TILE_IMAGE = pygame.image.load("tile.jpg")
MTILE_IMAGE = pygame.image.load("markedtile.jpg")
HOUSE1_IMAGE = pygame.image.load("house1.jpg")
HOUSE2_IMAGE = pygame.image.load("house2.jpg")
HOUSE3_IMAGE = pygame.image.load("house3.jpg")
TREE1_IMAGE  = pygame.image.load("tree1.jpg")
TREE2_IMAGE  = pygame.image.load("tree2.jpg")
UFO_IMAGE = pygame.image.load("ufo.jpg")
FLAG_IMAGE = pygame.image.load("flag.jpg")


TILE = pygame.transform.scale(TILE_IMAGE, (TILESIZE, TILESIZE))
MTILE = pygame.transform.scale(MTILE_IMAGE, (TILESIZE, TILESIZE))
HOUSE1 = pygame.transform.scale(HOUSE1_IMAGE, (TILESIZE, TILESIZE))
HOUSE2 = pygame.transform.scale(HOUSE2_IMAGE, (TILESIZE, TILESIZE))
HOUSE3 = pygame.transform.scale(HOUSE3_IMAGE, (TILESIZE, TILESIZE))
TREE1 = pygame.transform.scale(TREE1_IMAGE, (TILESIZE, TILESIZE))
TREE2 = pygame.transform.scale(TREE2_IMAGE, (TILESIZE, TILESIZE))
UFO = pygame.transform.scale(UFO_IMAGE, (TILESIZE, TILESIZE))
FLAG = pygame.transform.scale(FLAG_IMAGE, (TILESIZE, TILESIZE))


def draw_window(ufo, env):

    for i in range(env.width):
        for j in range(env.height):
            t = env.get_tile_type(i, j)
            if t == 1:
                WIN.blit(TREE1, (i*TILESIZE, j*TILESIZE))
            elif t == 2:
                WIN.blit(HOUSE1, (i*TILESIZE, j*TILESIZE))
            elif t == 3:
                WIN.blit(HOUSE2, (i*TILESIZE, j*TILESIZE))
            elif t == 4:
                WIN.blit(HOUSE3, (i*TILESIZE, j*TILESIZE))  
            elif t == 5:
                WIN.blit(TREE2, (i*TILESIZE, j*TILESIZE))     
            else:
                WIN.blit(TILE, (i*TILESIZE, j*TILESIZE))
    
        
    for (x, y) in ufo.tiles:
        WIN.blit(MTILE, (x*TILESIZE, y*TILESIZE))
        
    
    WIN.blit(FLAG, (env.goalx * TILESIZE, env.goaly * TILESIZE))        
    WIN.blit(UFO, (ufo.x * TILESIZE, ufo.y * TILESIZE))
        
    pygame.display.update()
    
    
    

def main():
    
    
    #  <------------   nastavení startu a cíle prohledávání !!!!!!!!!!
    env.set_start(0, 0)
    env.set_goal(9, 7)
    
    
    p, t = env.path_planner(algorithm="a_star")   # cesta pomocí path_planneru prostředí
    ufo.set_path(p, t)
    # ---------------------------------------------------
    
    
    clock = pygame.time.Clock()
    
    run = True
    go = False    
    
    while run:  
        
        clock.tick(FPS)
        

        # <---- reaktivní pohyb dokud nedojde do cíle 
        if (ufo.x != env.goalx) or (ufo.y != env.goaly):        
            #x, y = ufo.reactive_go(env)
            x, y = ufo.execute_path()
            
            if env.is_valid_xy(x, y):
                ufo.move(x, y)
            else:
                print('[', x, ',', y, ']', "wrong coordinate !")

        draw_window(ufo, env)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
    
    pygame.quit()    


if __name__ == "__main__":
    main()