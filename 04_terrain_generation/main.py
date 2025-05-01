import matplotlib.pyplot as plt
y = [1.0, 0.2, 0.5, 0.6, 0.2, 0.7, 0.8]


# teren je seznam realnych cisel mezi 0 a 1, 0.5 je hladina more
def plotterrain(t):

    fig, ax = plt.subplots()

    x = range(len(t))
    sea = [0.5 for i in range(len(t))]

    ax.fill_between(x, sea, color="turquoise")
    ax.fill_between(x, t, color="sandybrown")
    ax.axis("off")

    plt.show()


plotterrain(y)
