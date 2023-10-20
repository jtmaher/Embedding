import uuid
import networkx as nx
from util.graph_utils import hierarchy_pos


def struct_to_graph_impl(s, p, pa, G):
    node = uuid.uuid4()
    G.add_node(node, label=s.label)
    if p is not None:
        G.add_edge(p, node, label=pa)

    for k, v in s.attributes.items():
        struct_to_graph_impl(v, node, k, G)

    return


def struct_to_graph(s):
    G = nx.DiGraph()
    struct_to_graph_impl(s, None, None, G)
    return G


def draw_struct(s):
    opts = dict(node_size=1000, node_color='lightblue', font_size=10)
    G = struct_to_graph(s)
    pos = hierarchy_pos(G)
    nx.draw(G, pos, with_labels=False, **opts)
    labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos, labels=labels)

    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)