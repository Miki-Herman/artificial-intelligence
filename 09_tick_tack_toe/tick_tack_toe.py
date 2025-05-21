import numpy as np
import copy

# ----------- Pomocné Funkce -----------

# kreslí hru
def printgame(g):
    for r in g:
        pr = ""
        for i in r:
            if i == 0:
                pr += "."
            elif i == 1:
                pr += "x"
            else:
                pr += "o"
        print(pr)



# říká kdo vyhrál 0=nikdo, 1, 2
def whowon(g):

    # řádek
    if g[0][:] == [1, 1, 1] or g[1][:] == [1, 1, 1] or g[2][:] == [1, 1, 1]:
        return 1

    if g[0][:] == [2, 2, 2] or g[1][:] == [2, 2, 2] or g[2][:] == [2, 2, 2]:
        return 2

    # 1. sloupec
    if g[0][0] == g[1][0] == g[2][0] == 1:
        return 1

    if g[0][0] == g[1][0] == g[2][0] == 2:
        return 2

    # 2. sloupec
    if g[0][1] == g[1][1] == g[2][1] == 1:
        return 1

    if g[0][1] == g[1][1] == g[2][1] == 2:
        return 2


    # 3. sloupec
    if g[0][2] == g[1][2] == g[2][2] == 1:
        return 1

    if g[0][2] == g[1][2] == g[2][2] == 2:
        return 2


    # hlavní diagonála
    if g[0][0] == g[1][1] == g[2][2] == 1:
        return 1

    if g[0][0] == g[1][1] == g[2][2] == 2:
        return 2


    # hlavní anti-diagonála
    if g[0][2] == g[1][1] == g[2][0] == 1:
        return 1

    if g[0][2] == g[1][1] == g[2][0] == 2:
        return 2

    return 0


# vrací prázdná místa na šachovnici
def emptyspots(g):
    emp = []
    for i in range(3):
        for j in range(3):
            if g[i][j] == 0:
                emp.append((i, j))
    return emp


# ------------ Herní funkce ---------------
def ttt_move(game, my_player, other_player):

    new_game = copy.deepcopy(game)
    empty = emptyspots(new_game)

    # Pokud někdo vyhrál nebo je pole plný vrať stav
    if not empty or whowon(new_game) != 0:
        return new_game

    # Hledání nejlepšího tahu pomocí minmax
    best_score = float('-inf')
    best_move = None

    # Otestování každého prázdného pole
    for move in empty:
        row, col = move
        new_game[row][col] = my_player

        # Kalkulace skóre před zahráním dalšího tahu
        score = minimax(new_game, 0, False, my_player, other_player, float('-inf'), float('inf'))

        # Vrať zpět tah
        new_game[row][col] = 0

        # Aktualizace nejepšího tahu
        if score > best_score:
            best_score = score
            best_move = move

    # Uskutečnění nejlepšího tahu
    if best_move:
        row, col = best_move
        new_game[row][col] = my_player

    return new_game


def minimax(game, depth, is_maximizing, my_player, other_player, alpha, beta):

    # Kontrola koncových stavů
    winner = whowon(game)
    if winner == my_player:
        return 10 - depth  # Win (preferujeme rychlé hry)
    elif winner == other_player:
        return depth - 10  # Loss (preferujeme dlouhé hry v případě prohry)
    elif not emptyspots(game):
        return 0  # Draw

    # Maximalizace hráče (my_player)
    if is_maximizing:
        best_score = float('-inf')
        for move in emptyspots(game):
            row, col = move
            game[row][col] = my_player
            score = minimax(game, depth + 1, False, my_player, other_player, alpha, beta)
            game[row][col] = 0  # Vrátí zpět tah
            best_score = max(score, best_score)
            alpha = max(alpha, best_score)
            if beta <= alpha:
                break
        return best_score

    # Minimalizování hráče (other_player)
    else:
        best_score = float('inf')
        for move in emptyspots(game):
            row, col = move
            game[row][col] = other_player
            score = minimax(game, depth + 1, True, my_player, other_player, alpha, beta)
            game[row][col] = 0  # Vrátí zpět tah
            best_score = min(score, best_score)
            beta = min(beta, best_score)
            if beta <= alpha:
                break
        return best_score
