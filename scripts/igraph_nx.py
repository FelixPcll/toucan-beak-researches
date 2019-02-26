import networkx as nx
from igraph import Graph as Graph

def igraph2nx(g):

	gnx = nx.Graph()
	node_att = [(node.index, node.attributes()) for node in g.vs]
	gnx.add_nodes_from(node_att)
	edge_att = [(edge.source, edge.target, edge.attributes()) for edge in g.es]
	gnx.add_edges_from(edge_att)
	
	return gnx
	
def nx2igraph(gnx, node_attrs_to_keep=None, edge_attrs_to_keep=None):

	nodes = list(gnx.nodes(data=True))
	node_names, node_attrs = zip(*nodes)
	node_name_to_index = dict(zip(node_names, range(len(node_names))))
	edges = list(gnx.edges(data=True))
	source_nodes, target_nodes, edge_attrs = zip(*edges)
	source_nodes = [node_name_to_index[node] for node in source_nodes]
	target_nodes = [node_name_to_index[node] for node in target_nodes]
				
	g = Graph(n=len(node_names), edges=list(zip(source_nodes, target_nodes)), directed=True)
	
	if node_attrs_to_keep is None:
		if len(node_attrs)>0:
			for attrs in node_attrs[0]:
				g.vs[attrs] = [node[attrs] for node in node_attrs]
	else:
		for attrs in node_attrs_to_keep:
			g.vs[attrs] = [node[attrs] for node in node_attrs]		
	if edge_attrs_to_keep is None:
		if len(edge_attrs)>0:
			for attrs in edge_attrs[0]:
				g.es[attrs] = [edge[attrs] for edge in edge_attrs]		
	else:
		for attrs in edge_attrs_to_keep:
			g.es[attrs] = [edge[attrs] for edge in edge_attrs]		

	g.vs['nx_name'] = node_names
	
	return g