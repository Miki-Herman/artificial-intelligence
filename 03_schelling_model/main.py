from schelling_class import SchellingModel

model = SchellingModel(size=20, empty_prob=0.4 , type_prob=0.3, threshold=0.3)
model.simulate(max_steps=50, speed=0.5)
