import random
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ==============================
# Parameters
# ==============================

NUM_NEURONS = 20
STEPS = 400
RADIUS = 0.25
DRIFT = 0.01
NOISE = 0.05

POKE_STEP = 50
POKE_NEURON = 5
POKE_VALUE = 3

PREDATOR_INTERVAL = 100
PREDATOR_SPEED = 0.05
PREDATOR_RADIUS = 0.05

SOLID = "solid"
LIQUID = "liquid"
PLASMA = "plasma"

# ==============================
# Neuron
# ==============================

class Neuron:
    def __init__(self):
        self.x = random.random()
        self.y = random.random()
        self.state = random.randint(1, 3)
        self.phase = LIQUID
        self.memory = self.state

    def distance(self, other):
        return math.hypot(self.x - other.x, self.y - other.y)

neurons = [Neuron() for _ in range(NUM_NEURONS)]

# ==============================
# Tension Spider
# ==============================

class TensionSpider:
    def __init__(self, solid_thresh=0.2, plasma_thresh=0.8):
        self.solid_thresh = solid_thresh
        self.plasma_thresh = plasma_thresh

    def apply(self, neuron, neighbors):
        if not neighbors:
            neuron.phase = PLASMA
        else:
            values = [n.state for n in neighbors]
            variance = sum((v - sum(values)/len(values))**2 for v in values)/len(values)
            if variance < self.solid_thresh:
                neuron.phase = SOLID
            elif variance > self.plasma_thresh:
                neuron.phase = PLASMA
            else:
                neuron.phase = LIQUID

def plasma_mutate_spider(neurons, spider):
    plasma_neurons = [n for n in neurons if n.phase == PLASMA]
    if plasma_neurons:
        spider = TensionSpider(
            solid_thresh=random.uniform(0.1,0.4),
            plasma_thresh=random.uniform(0.6,0.9)
        )
        print(f">>> Plasma mutated spider: solid={spider.solid_thresh:.2f}, plasma={spider.plasma_thresh:.2f}")
    return spider

# ==============================
# Predator
# ==============================

class Predator:
    def __init__(self):
        self.x = random.random()
        self.y = random.random()
        self.neurons_eaten = 0
        self.hunger_timer = 0

    def move(self):
        tension = self.hunger_timer / 50.0
        speed = PREDATOR_SPEED * (1 + tension*2)
        angle = random.uniform(0, 2*math.pi)
        if tension > 1.0 and random.random() < 0.2:
            self.x = random.random()
            self.y = random.random()
        else:
            self.x = (self.x + speed*math.cos(angle)) % 1.0
            self.y = (self.y + speed*math.sin(angle)) % 1.0
        self.hunger_timer += 1

    def eat(self):
        self.neurons_eaten += 1
        self.hunger_timer = 0

predator = None
predator_timer = PREDATOR_INTERVAL

# ==============================
# Cluster detection
# ==============================

def detect_clusters(neurons, radius=RADIUS):
    visited = set()
    clusters = []
    for i, n in enumerate(neurons):
        if i in visited:
            continue
        cluster = {i}
        to_check = [i]
        while to_check:
            idx = to_check.pop()
            visited.add(idx)
            for j, m in enumerate(neurons):
                if j not in visited and n.state == m.state and n.distance(m) < radius:
                    cluster.add(j)
                    to_check.append(j)
        clusters.append(cluster)
    return clusters

# ==============================
# Visualization
# ==============================

fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(0,1)
ax.set_ylim(0,1)
scat = ax.scatter([], [], s=200)
pred_plot, = ax.plot([], [], 'rx', markersize=10)

spider = TensionSpider()

# ==============================
# Animation update
# ==============================

def update(frame):
    global neurons, predator, predator_timer, spider
    step = frame

    # Poke neuron
    if step == POKE_STEP:
        neurons[POKE_NEURON].state = POKE_VALUE
        neurons[POKE_NEURON].phase = LIQUID
        print(f">>> POKE at step {step}: neuron {POKE_NEURON} -> {POKE_VALUE}")

    # Spawn predator
    if predator_timer <= 0:
        predator = Predator()
        predator_timer = PREDATOR_INTERVAL
        print(f">>> Predator entered at step {step}")
    else:
        predator_timer -= 1

    # Neighborhood discovery
    neighborhoods = []
    for n in neurons:
        neighbors = [m for m in neurons if n.distance(m) < RADIUS and m is not n]
        neighborhoods.append(neighbors)

    # Phase updates
    for i, n in enumerate(neurons):
        spider.apply(n, neighborhoods[i])

    # Plasma can mutate spider
    spider = plasma_mutate_spider(neurons, spider)

    # State updates
    for i, n in enumerate(neurons):
        neighbors = neighborhoods[i]
        if n.phase == PLASMA:
            n.state = random.randint(1,3)
        elif n.phase == LIQUID and neighbors:
            avg = sum(m.state for m in neighbors)/len(neighbors)
            n.state = int(round(avg)) + random.choice([-1,0,1])
            n.state = max(1,min(3,n.state))
        elif n.phase == SOLID:
            n.state = n.memory
        n.memory = n.state

    # Spatial self-organization
    for i, n in enumerate(neurons):
        neighbors = neighborhoods[i]
        for m in neighbors:
            dx = m.x - n.x
            dy = m.y - n.y
            if n.state == m.state:
                n.x += DRIFT * dx
                n.y += DRIFT * dy
            else:
                n.x -= DRIFT * dx
                n.y -= DRIFT * dy
        n.x = (n.x + NOISE*(random.random()-0.5)) % 1.0
        n.y = (n.y + NOISE*(random.random()-0.5)) % 1.0

    # Predator interaction
    if predator:
        predator.move()
        remaining_neurons = []
        for n in neurons:
            if n.distance(predator) < PREDATOR_RADIUS:
                predator.eat()
            else:
                remaining_neurons.append(n)
        neurons = remaining_neurons

    # Draw neurons
    colors = []
    xs = []
    ys = []
    for n in neurons:
        xs.append(n.x)
        ys.append(n.y)
        if n.phase == SOLID:
            colors.append('blue')
        elif n.phase == LIQUID:
            colors.append('green')
        else:
            colors.append('red')
    scat.set_offsets(list(zip(xs, ys)))
    scat.set_color(colors)

    # Draw predator
    if predator:
        pred_plot.set_data(predator.x, predator.y)
    else:
        pred_plot.set_data([], [])

    ax.set_title(f"Step {step} | Neurons: {len(neurons)}")
    return scat, pred_plot

ani = animation.FuncAnimation(fig, update, frames=STEPS, interval=100, blit=True)
plt.show()
