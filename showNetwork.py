import networkx as nx
import matplotlib.pyplot as plt

# Створюємо граф
G = nx.DiGraph()

# Додаємо вузли
input_nodes = ['Input']
hidden_nodes_layer1 = []
hidden_nodes_layer2 = []
output_nodes = ['Output']

for i in range(5):
    hidden_nodes_layer1.append(f'H1_{i+1}')
    G.add_node(f'H1_{i+1}')

for i in range(5):
    hidden_nodes_layer2.append(f'H2_{i + 1}')
    G.add_node(f'H2_{i + 1}')

# Додаємо зв'язки між вузлами
for input_node in input_nodes:
    for hidden_node in hidden_nodes_layer1:
        G.add_edge(input_node, hidden_node)

for hidden_node_1 in hidden_nodes_layer1:
    for hidden_node_2 in hidden_nodes_layer2:
        G.add_edge(hidden_node_1, hidden_node_2)

for hidden_node in hidden_nodes_layer2:
    for output_node in output_nodes:
        G.add_edge(hidden_node, output_node)

# Визначаємо позиції вузлів для відображення на графіку
pos = {}
pos.update((node, (0, 400)) for node in input_nodes)  # Позиції вхідних вузлів
pos.update((node, (25, i*200)) for i, node in enumerate(hidden_nodes_layer1))  # Позиції вузлів прихованого шару
pos.update((node, (50, i*200)) for i, node in enumerate(hidden_nodes_layer2))  # Позиції вузлів прихованого шару
pos.update((node, (75, 400)) for node in output_nodes)  # Позиції вихідних вузлів

# Відображаємо граф
nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=300)
nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='->', arrowsize=10)
nx.draw_networkx_labels(G, pos, font_size=5, font_color='black', verticalalignment='center')

# Зберігаємо графік
plt.savefig('neuron_network.png', dpi=300)


