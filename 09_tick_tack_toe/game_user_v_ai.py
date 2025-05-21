from tick_tack_toe import *

game = [[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]]

print("Začíná hra piškvorky!")
print("Hrajete jako 'x' (hráč 1), počítač je 'o' (hráč 2).")
printgame(game)

while True:
    # Hráčův tah
    while True:
        try:
            move = input("Zadej svůj tah ve formátu 'řádek sloupec' (např. 0 2): ")
            row, col = map(int, move.strip().split())
            if game[row][col] == 0:
                game[row][col] = 1
                break
            else:
                print("Toto pole je již obsazeno. Zkus jiné.")
        except:
            print("Neplatný vstup. Zadej dvě čísla mezi 0 a 2.")

    print("Herní deska po tvém tahu:")
    printgame(game)

    if whowon(game) == 1:
        print("Vyhrál jsi!")
        break
    elif not emptyspots(game):
        print("Remíza.")
        break

    # Tah počítače
    print("Počítač hraje...")
    game = ttt_move(game, 2, 1)
    printgame(game)

    if whowon(game) == 2:
        print("Počítač vyhrál!")
        break
    elif not emptyspots(game):
        print("Remíza.")
        break