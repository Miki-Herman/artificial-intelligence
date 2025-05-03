import random

# vždy kooperuje
def always_cooperate(my_history, other_history):
    return 0


# náhodná odpověď
def random_answer(my_history, other_history):
    p = random.random()
    if p < 0.5:
        return 1

    return 0

# oko za oko -> pokud druhý vezeň zradí, tak následující tah ho také zradí
def tit_for_tat(my_histroy, other_history):

    if len(my_histroy) == 0:
        return 0

    elif other_history[-1] == 1:
        return 1

    else:
        return 0

# vždy zradí
def always_defect(my_history, other_history):
    return 1

# zradí po dvou zradách
def tit_for_two_tats(my_history, other_history):
    if len(my_history) < 2:
        return 0
    elif other_history[-1] == 1 and other_history[-2] == 1:
        return 1
    else:
        return 0

# spolupracuje, dokud soupeř nezradí – pak už vždy zrazuje
def grim_trigger(my_history, other_history):
    return 0 if 1 not in other_history else 1

# Náhodná strategie s předsudkem (80 % spolupráce)
def biased_random(my_histroy, other_history):
    return 0 if random.random() < 0.8 else 1

# Pavlov (Win-Stay, Lose-Shift)
def pavlov(my_history, other_history):
    if len(my_history) == 0:
        return 0
    if my_history[-1] == other_history[-1]:  # pokud výsledek byl shodný, pokračuj
        return my_history[-1]
    else:
        return 1 - my_history[-1]  # změň tah

# Strategie testuje reakce soupeře
def probing(my_history, other_history):
    if len(my_history) < 3:
        return [0, 1, 1][len(my_history)]  # testuje reakci
    elif other_history[1] == 0:
        return 1  # pokud soupeř byl příliš důvěřivý, zrazuje
    else:
        return other_history[-1]  # jinak napodobuje

# Zrazuje pouze 2 kola poté co byl zrazen
def soft_grudger(my_history, other_history):
    punishment = 2
    if 1 in other_history:
        last_betray = len(other_history) - 1 - other_history[::-1].index(1)
        if len(other_history) - last_betray <= punishment:
            return 1
    return 0

# Zrazuje střídavě
def alternator(my_history, other_history):
    return len(my_history) % 2

# Oko za oko s procentem odpuštění
def generous_tit_for_tat(my_history, other_history):
    if len(other_history) == 0:
        return 0
    if other_history[-1] == 1:
        return 0 if random.random() < 0.1 else 1
    return 0