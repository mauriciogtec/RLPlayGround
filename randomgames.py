import numpy as np

class Node:
    nodeid = 0
    "Node in a tree" 
    def __init__(self, parent):
        self.parent = parent
        self.depth = (0 if self.is_root() else parent.depth + 1)
        self.children = []
        self.nodeid = Node.nodeid
        Node.nodeid += 1
        
    def is_root(self):
        return self.parent == self
    
    def set_nodeid(value=0):
        Node.nodeid = value

class ActionNode(Node):
    "Type of node where a decision must be taken"
    def __init__(self, parent, player):
        super().__init__(parent)
        self.player = player
        self.decision = np.nan
        self.value = (np.nan, np.nan)
    
    def reproduce_actions(self, lam, minactions, maxactions):
        nchildren = np.clip(np.random.poisson(lam), minactions, maxactions)
        next_player = (self.player + 1) % 2
        self.children = [ActionNode(self, next_player) for _ in range(nchildren)]
        
    def reproduce_payoffs(self, lam, minpayoffs, maxpayoffs, ties):
        nchild = np.clip(np.random.poisson(lam), minpayoffs, maxpayoffs) # at least one
        opts = [-1, 0, 1] if ties else [-1, 1]
        payoffs0 = [np.random.choice(opts) for _ in range(nchild)]
        self.children = [PayOffNode(self, (-v, v)) for v in payoffs0]
        
    def __repr__(self):
        decision_str = "" if np.isnan(self.decision) else "selected: {}".format(self.decision)
        value_str = "" if np.isnan(self.value[0]) else "value: ({}, {})".format(self.value[0], self.value[1])
        return "Action(id: {}, player: {}, {}, {})".\
            format(self.nodeid, self.player, decision_str, value_str)
    
class GameRootNode(ActionNode):
    "Regular action node but self refential"
    def __init__(self):
        parent = self
        player = 0
        super().__init__(parent, player)    
    
class PayOffNode(Node):
    "A node with a payoff value"
    def __init__(self, parent, payoff):
        assert len(payoff) == 2
        super().__init__(parent)
        self.payoff = payoff
        parent.children = [self]
        self.selected = np.nan
        
    def __repr__(self):
        return "PayOff(id: {}, value: ({:d}, {:d}))".format(self.nodeid, *self.payoff)

class TwoPlayerGame:
    def __init__(self, 
                 root=GameRootNode(), 
                 lam=2., maxdepth=10,
                 generate=True,
                 minactions=0, 
                 maxactions=99, 
                 minpayoffs=2,
                 maxpayoffs=99,
                 ties=False):
        Node.set_nodeid(1)
        self.root = root
        self.lam = lam
        self.minactions=minactions
        self.maxactions=maxactions
        self.minpayoffs=minpayoffs
        self.maxpayoffs=maxpayoffs
        self.maxdepth = maxdepth
        self.ties = ties
        self.solved = False
        self.num_nodes = 1
        
        if generate:
            self.generate()
      
    def generate(self):      
        "Randomly reproduces nodes until maxdepth or no descendents found"
        # while a branch has a non-payoff node
        nonpayoff = [self.root]
        while len(nonpayoff) > 0:
            # take one nonterminal node and reproduce
            node = nonpayoff.pop()
            if node.depth < self.maxdepth - 1:
                node.reproduce_actions(self.lam, minactions=self.minactions, maxactions=self.maxactions)
                
                # max_depth reached or no children then attach payoff nodes, else add to nonpayoff
                if len(node.children) > 0:     
                    for child in node.children:
                        nonpayoff.append(child)
                        self.num_nodes += 1
            else:
                # spawn a generation of payoffs
                node.reproduce_payoffs(self.lam, self.minpayoffs, self.maxpayoffs, self.ties)
                self.num_nodes += len(node.children)
    
    def branch(self, node):
        "returns a pointer to the same fields of the tree but different root"
        return TwoPlayerGame(root=node, generate=False)
    
    def solve(self):
        "uses backward induction (dynamic programming) to find a Nash equlibrium"
        node = self.root      
        if isinstance(node, ActionNode):
            player = node.player
            children = node.children
            child_value = []
            for i, child in enumerate(children):
                if isinstance(child, PayOffNode):
                    child_value.append(child.payoff)
                else:
                    child_subgame = self.branch(child)
                    child_subgame.solve()
                    subgame_value = child_subgame.root.value
                    child_value.append(subgame_value)
    
            node.decision = np.argmax([x[player] for x in child_value])
            node.value = child_value[node.decision]
        self.solved = True
        
    def print_solution_path(self):
        "Prints selected nodes only"
        assert self.solved
        node = self.root
        s = ('  ' * node.depth) + str(node) + '\n'
        while not isinstance(node, PayOffNode):
            node = node.children[node.decision]
            s += (':-' * node.depth) + str(node) + '\n'
        if node.payoff[0] > node.payoff[1]:
            s += "Player 0 wins"
        elif node.payoff[1] > node.payoff[0]:
            s += "Player 1 wins"
        else:
            s += "Player 0 and 1 draw"
        print(s)

    def __repr__(self):
        return "Tree with {:d} nodes".format(self.num_nodes) 

    def __str__(self):
        "Depth-First printing"
        node = self.root
        strout = (':-' * node.depth) + str(node) + '\n'
        if isinstance(node, PayOffNode):
            return strout
        else:
            children = node.children
            for child in children:
                strout += self.branch(child).__str__()
            return strout
        
    def valuemap(self, playerid):
        # we do BFS for saving in a dict all values
        assert self.solved
        valuemap = dict()
        curr=self.root
        valuemap[curr.nodeid]=curr.value[playerid]
        to_visit=curr.children.copy()
        while len(to_visit) > 0:
            curr = to_visit.pop()
            for x in curr.children:
                to_visit.append(x)
            if isinstance(curr, PayOffNode):
                valuemap[curr.nodeid]=curr.payoff[playerid]
            else:
                valuemap[curr.nodeid]=curr.value[playerid]   
 
            
        return valuemap