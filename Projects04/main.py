class SemanticNetwork:
    def __init__(self):
        self.network = ()
        
    def add_node(self, node):
        if node not  in self.network:
            self.network[node] = {'type_of': [], 'has_characteristic': []}
            
    def add_relationship(self, parent, child, relationship):
        if parent in self.network and child in self.network:
            if relationship == 'is a type of':
                self.network[parent]['type_of'].append(child)
            elif relationship == 'has charachteristic':
                self.network[parent]['has_characteristic'].append(child)
                
    def dispaly_network(self):
        for node, relation in self.network.items():
            print(f'Node : {node}')
            for relation, children in relation.items():
                for child in children:
                    print(f'\t{relation} : {child}')
                    
# create the semantic network
network = SemanticNetwork()
