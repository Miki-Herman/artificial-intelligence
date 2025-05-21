from tick_tack_toe import *
import time

def play_ai_vs_ai(num_games=1, delay=0.5):
    """
    Simulate a game between two AI players.

    Args:
        num_games: Number of games to play
        delay: Time delay between moves (in seconds) for better visualization
    """
    results = {"AI1 (X) wins": 0, "AI2 (O) wins": 0, "Draws": 0}

    for game_num in range(num_games):
        print(f"\n=== HRA {game_num + 1} ===")
        game = [[0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]]

        print("Počáteční stav:")
        printgame(game)

        move_count = 0

        while True:
            move_count += 1

            # AI 1 tah (X - hráč 1)
            print(f"\nTah {move_count}: AI 1 (X) hraje...")
            time.sleep(delay)
            game = ttt_move(game, 1, 2)
            printgame(game)

            # Zkontrolovat jestli AI 1 vyhrál nebo je remíza
            if whowon(game) == 1:
                print("AI 1 (X) vyhrál!")
                results["AI1 (X) wins"] += 1
                break
            elif not emptyspots(game):
                print("Remíza!")
                results["Draws"] += 1
                break

            move_count += 1

            # AI 2 tah (O - hráč 2)
            print(f"\nTah {move_count}: AI 2 (O) hraje...")
            time.sleep(delay)
            game = ttt_move(game, 2, 1)
            printgame(game)

            # Zkontrolovat jestli AI 2 vyhrál nebo je remíza
            if whowon(game) == 2:
                print("AI 2 (O) vyhrál!")
                results["AI2 (O) wins"] += 1
                break
            elif not emptyspots(game):
                print("Remíza!")
                results["Draws"] += 1
                break

    # Vypsat výsledky
    print("\n=== VÝSLEDKY ===")
    print(f"Počet her: {num_games}")
    print(f"AI1 (X) výhry: {results['AI1 (X) wins']}")
    print(f"AI2 (O) výhry: {results['AI2 (O) wins']}")
    print(f"Remízy: {results['Draws']}")
    print(f"AI1 výherní poměr: {results['AI1 (X) wins']/num_games:.2%}")
    print(f"AI2 výherní poměr: {results['AI2 (O) wins']/num_games:.2%}")
    print(f"Remízy poměr: {results['Draws']/num_games:.2%}")

    return results

# Spustit hru mezi dvěma AI hráči
print("Spouštím souboj dvou AI hráčů v piškvoркách...")
play_ai_vs_ai(num_games=3, delay=0.5)  # Můžete změnit počet her a zpoždění
