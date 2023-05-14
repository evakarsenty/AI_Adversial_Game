import copy
import itertools
import math
import random
import networkx as nx
from Simulator import Simulator

IDS = [111111111, 222222222]


class Agent:
    def __init__(self, initial_state, player_number):
        self.ids = IDS
        self.agent = UCTAgent(initial_state, player_number)

    def act(self, state):
        return self.agent.act(state)

    def Greddy_action(self, node):
            """
                UCT Method: Assumes that all childs are expanded
                Implements given policy
            """
            if node.getExpanded() == 0:
                return random.choice(node.children)
            en = min(1, (0.4 * len(node.children))/(1**2 * node.getExpanded()))
            return node.best_child[1] if random.random() > en else random.choice(node.children)


class UCTAgent:
    def __init__(self, initial_state, player_number):
        self.ids = IDS
        self.impass = self.init_impass(initial_state)
        self.initial_state = initial_state
        self.dimensions = len(self.initial_state['map']), len(self.initial_state['map'][0])
        self.simulator = Simulator(initial_state)
        self.base_state = copy.deepcopy(initial_state)
        self.player_nb = player_number
        self.my_taxis = []
        self.score = {'player 1': 0, 'player 2': 0}
        for taxi_name, taxi in initial_state['taxis'].items():
            if taxi['player'] == player_number:
                self.my_taxis.append(taxi_name)
        self.initial_data_distances = DistFunctions(initial_state)
        self.UCT_tree = self.build_UCT_tree(self.initial_state)
        self.root_node = self.UCT_alg(self.UCT_tree)
        self.G_node = copy.deepcopy(self.root_node)

    def init_impass(self, initial):
        impass = ()
        row = len(initial['map'][0])
        col = len(initial['map'])
        for i in range(row):
            for j in range(col):
                if initial['map'][i][j] == 'I':
                    impass = impass + ((i, j),)
        return impass

    def build_UCT_tree (self, initial_state):
        taxi_actions = []
        for taxi_name in self.my_taxis:
            self.base_state['taxis'][taxi_name]['max_capacity'] = self.initial_state['taxis'][taxi_name]['capacity']
            taxi_actions.append(('initial', taxi_name, initial_state['taxis'][taxi_name]['location']))
        initial = tuple(taxi_actions)
        root_node = UCTNode(initial, None, [], 0, 0, None)
        all_actions = []
        for taxi_name in self.my_taxis:
            taxi_actions = []
            x, y = initial_state['taxis'][taxi_name]['location']
            if x - 1 >= 0 and initial_state['map'][x - 1][y] != self.impass:
                taxi_actions.append(('move', taxi_name, (x - 1, y)))
            if x + 1 < self.dimensions[1] and initial_state['map'][x + 1][y] != self.impass:
                taxi_actions.append(('move', taxi_name, (x + 1, y)))
            if y - 1 >= 0 and initial_state['map'][x][y - 1] != self.impass:
                taxi_actions.append(('move', taxi_name, (x, y - 1)))
            if y + 1 < self.dimensions[0] and initial_state['map'][x][y + 1] != self.impass:
                taxi_actions.append(('move', taxi_name, (x, y + 1)))

            for passenger_name in self.initial_state['passengers'].keys():
                taxi_actions.append(('pick up', taxi_name, passenger_name))
                taxi_actions.append(('drop off', taxi_name, passenger_name))
            taxi_actions.append(('wait', taxi_name))
            all_actions.append(taxi_actions)
        all_possible_actions = list(itertools.product(*all_actions))
        all_actions = self.check_actions(initial_state, all_possible_actions, self.player_nb)
        for action in all_actions:
            child_node = UCTNode(action, root_node, [], 0, 0, None)
            root_node.children.append(child_node)
        return root_node

    def UCT_alg(self, UCT_tree):
        """
        Implementation of the UCT algorithm while using the 4 steps: selection, expansion, simulate, backpropagation
        """
        current_node = UCT_tree
        state = copy.deepcopy(self.initial_state)
        selected_node = self.selection(UCT_tree, current_node)
        next_node = self.apply_action(state, selected_node.action, self.player_nb)
        if selected_node.visit_count == 0:
            simulation_is_possible = self.test_expansion_simulation(selected_node)
            if simulation_is_possible:
                simulation_result = [self.simulation(next_node), selected_node]
                self.backpropagation(simulation_result)
        else:
            if not selected_node.children:
                expansion_is_possible = self.test_expansion_simulation(selected_node)
                if expansion_is_possible:
                    self.expansion(UCT_tree, selected_node, next_node)
        return UCT_tree

    def selection(self, UCT_tree, parent_node):
        """
        Selects the next leaf node to expand by traversing the tree and applying the UCB1 formula
        """
        best_node = None
        best_UCB1 = -float('inf')
        for child in parent_node.children:
            if child.visit_count == 0:
                UCB1 = float('inf')
            else:
                UCB1 = (child.total_score / child.visit_count) + math.sqrt(2*math.log(UCT_tree.visit_count) / child.visit_count)
            if UCB1 > best_UCB1:
                best_UCB1 = UCB1
                best_node = child
        return best_node

    def expansion(self, UCT_tree, parent_node, parent_state):
        """
        Expands the selected leaf node by adding its children to the tree.
        """
        all_actions = self.build_actions(parent_state)
        for action in all_actions:
            child_node = UCTNode(action, parent_node, [], 0, 0, None)
            parent_node.children.append(child_node)

    def simulation(self, state):
        """
        Simulates the game from the current state and returns the result of the simulation.
        """
        current_state = copy.deepcopy(state)
        self.score = {'player 1': 0, 'player 2': 0}
        count = 0
        while count != 300:
            all_actions = self.build_actions(current_state)
            next_action = self.choose_action(current_state, all_actions)
            next_state = self.apply_action(current_state, next_action, self.player_nb)
            count += 1
        return self.score[f"player {self.player_nb}"]

    def backpropagation(self, simulation_result):
        """
        Backpropagates the result of the simulation up the tree and updates the statistics of each visited node.
        """
        score, node = simulation_result[0], simulation_result[1]
        while node is not None:
            node.visit_count += 1
            node.total_score += score
            best_score = float('-inf')
            best_action = None
            for child in node.children:
                if child.total_score > best_score:
                    best_score = child.total_score
                    best_action = child.action
            node.best_action = best_action
            score = node.total_score
            node = node.parent

    def act(self, state):
        best_action = self.G_node.best_action
        if best_action is None:
            taxi_positions = [state['taxis'][taxi]['location'] for taxi in self.my_taxis]
            initial = ('initial', self.my_taxis, taxi_positions)
            root_node = UCTNode(initial, None, [], 0, 0, None)
            all_actions = self.build_actions(state)
            root_node.children = all_actions
            UCT_T= root_node
            best_action = self.choose_action(state, UCT_T.children)
            return best_action
        else:
            children = self.G_node.children
            for child in children:
                if child.action == best_action:
                    self.G_node = children[children.index(child)]
                    break
            return best_action

    def test_expansion_simulation(self, node):
        """
        This function tests whether or not a given node is a valid candidate for expansion
        or simulation in the UCT algorithm
        """
        count = 0
        for taxi_action in node.action:
            if node.parent.parent is None:
                if taxi_action[0] == 'move' and taxi_action[2] == node.parent.action[count][2]:
                    return False
            elif taxi_action == node.parent.action[count]:
                return False
            count += 1
        return True

    def in_bound(self, row, col, options):
        # moves=["move up","move down","move right","move left"]
        if row < self.dimensions[1] - 1:
            options[1] = (row + 1, col)  # down
        if row > 0:
            options[0] = (row - 1, col)  # up
        if col < self.dimensions[0] - 1:
            options[2] = (row, col + 1)  # right
        if col > 0:
            options[3] = (row, col - 1)  # left

    def next_pass(self, state, taxi_name, passenger):
        closest = {'location': None, 'name': None, 'distance': float('inf')}
        for passenger_name in state['passengers']:
            distance = self.initial_data_distances.get_distances(self.initial_data_distances.shortest_path_distances,
                                                                 state['taxis'][taxi_name]['location'],
                                                                 state['passengers'][passenger_name]['location'])
            if distance < closest['distance'] and passenger_name not in passenger:
                closest.update({'location': state['passengers'][passenger_name]['location'],
                                'name': passenger_name,
                                'distance': distance
                                })
        return closest['location'], closest['name']

    def update_passenger_loc(self, state, t_n):
        pass_dest = None
        for pass_n in state['passengers']:
            if state['passengers'][pass_n]['location'] == t_n:
                pass_dest = state['passengers'][pass_n]['destination']
                break
        return pass_dest

    def check_actions(self, state, all_possible_actions, player):
        all_actions = []
        for possible_action in all_possible_actions:
            if self.check_if_action_legal(state, possible_action, player):
                all_actions.append(possible_action)
        return all_actions

    def apply_action(self, state, action, player):
        for atomic_action in action:
            self._apply_atomic_action(state, atomic_action, player)
        state['turns to go'] -= 1
        return state

    def _apply_atomic_action(self, state, atomic_action, player):
        """
        Apply an atomic action to the state
        Derivation of the Simulatior function
        """
        taxi_name = atomic_action[1]
        if atomic_action[0] == 'move':
            state['taxis'][taxi_name]['location'] = atomic_action[2]
            return
        elif atomic_action[0] == 'pick up':
            passenger_name = atomic_action[2]
            state['taxis'][taxi_name]['capacity'] -= 1
            state['passengers'][passenger_name]['location'] = taxi_name
            return
        elif atomic_action[0] == 'drop off':
            passenger_name = atomic_action[2]
            state['taxis'][taxi_name]['capacity'] += 1
            self.score[f"player {player}"] += self.initial_state['passengers'][passenger_name]['reward']
            del state['passengers'][passenger_name]
            return
        elif atomic_action[0] == 'wait':
            return
        else:
            raise NotImplemented

    def choose_action(self, state, actions):
        """
        This function takes in the current game state and a list of possible actions as input,
        and returns on of these actions randomly
        """
        flag = False
        for i in range(300):
            final_action = []
            chosen_pass = []
            for action in actions.values():
                for act in action:
                    if act[0] == "drop off":
                        final_action.append(act)
                        break
                    if act[0] == "pick up":
                        final_action.append(act)
                        chosen_pass.append(act[2])
                        break
                else:
                    if flag:
                        final_action.append(random.choice(list(action)))
                    else:
                        final_action.append(self.calculate_best_action(state, action, chosen_pass))
            final_action = tuple(final_action)
            if self.check_if_action_legal(state, final_action, self.player_nb):
                return final_action
            else:
                flag = True
        t_act = []
        for taxi_name in self.my_taxis:
            t_act.append(('wait', taxi_name))
        t_act = tuple(t_act)
        return t_act

    def calculate_best_action(self, current_state, taxis_act, passenger):
        max_d = self.initial_data_distances.max_d
        best_action = None
        max_points = float('-inf')
        pass_name = None
        closest_pass_name = None
        for act in taxis_act:
            points = 0
            if act[0] == 'move':
                taxi_loc = act[2]
                if current_state['taxis'][act[1]]['capacity'] == self.base_state['taxis'][act[1]]['max_capacity']:
                    closest_pass_loc, closest_pass_name = self.next_pass(current_state,
                                                                         act[1], passenger)
                    points = max_d - self.initial_data_distances.get_distances(self.initial_data_distances.shortest_path_distances,
                                                                               taxi_loc, closest_pass_loc)
                else:
                    passenger_destination_location = self.update_passenger_loc(current_state, act[1])
                    points = max_d - self.initial_data_distances.get_distances(self.initial_data_distances.shortest_path_distances,
                                                                               taxi_loc, passenger_destination_location)
            elif act[0] == 'wait':
                if len(current_state['passengers'].keys()) > 0:
                    points = -max_d
                else:
                    points = max_d
            if points > max_points:
                max_points = points
                best_action = act
                pass_name = closest_pass_name
        passenger.append(pass_name)
        return best_action

    def check_if_action_legal(self,state, action, player):
        """
        Derivation of the Simulator function
        """
        def _is_move_action_legal(move_action, player):
            taxi_name = move_action[1]
            if taxi_name not in state['taxis'].keys():
                # logging.error(f"Taxi {taxi_name} does not exist!")
                return False
            if player != state['taxis'][taxi_name]['player']:
                # logging.error(f"Taxi {taxi_name} does not belong to player {player}!")
                return False
            l1 = state['taxis'][taxi_name]['location']
            l2 = move_action[2]
            if l2 not in self.neighbors(l1):
                # logging.error(f"Taxi {taxi_name} cannot move from {l1} to {l2}!")
                return False
            return True

        def _is_pick_up_action_legal(pick_up_action, player):
            taxi_name = pick_up_action[1]
            passenger_name = pick_up_action[2]
            # check same position
            if player != state['taxis'][taxi_name]['player']:
                return False
            if state['taxis'][taxi_name]['location'] != state['passengers'][passenger_name]['location']:
                return False
            # check taxi capacity
            if state['taxis'][taxi_name]['capacity'] <= 0:
                return False
            return True

        def _is_drop_action_legal(drop_action, player):
            taxi_name = drop_action[1]
            passenger_name = drop_action[2]
            # check same position
            if player != state['taxis'][taxi_name]['player']:
                return False
            if state['taxis'][taxi_name]['location'] != state['passengers'][passenger_name]['destination']:
                return False
            # check passenger is in the taxi
            if state['passengers'][passenger_name]['location'] != taxi_name:
                return False
            return True

        def _is_action_mutex(global_action):

            assert type(global_action) == tuple, "global action must be a tuple"
            # one action per taxi
            if len(set([a[1] for a in global_action])) != len(global_action):
                return True
            # pick up the same person
            pick_actions = [a for a in global_action if a[0] == 'pick up']
            if len(pick_actions) > 1:
                passengers_to_pick = set([a[2] for a in pick_actions])
                if len(passengers_to_pick) != len(pick_actions):
                    return True
            return False

        all_passengers = state['passengers'].keys()
        players_taxis = [taxi for taxi in state['taxis'].keys() if state['taxis'][taxi]['player'] == player]

        if len(action) != len(players_taxis):
            # logging.error(f"You had given {len(action)} atomic commands, while you control {len(players_taxis)}!")
            return False
        for atomic_action in action:
            # trying to act with a taxi that is not yours
            if atomic_action[1] not in players_taxis:
                # logging.error(f"Taxi {atomic_action[1]} is not yours!")
                return False
            # illegal move action
            if atomic_action[0] == 'move':
                if not _is_move_action_legal(atomic_action, player):
                    # logging.error(f"Move action {atomic_action} is illegal!")
                    return False
            # illegal pick action
            elif atomic_action[0] == 'pick up':
                if not _is_pick_up_action_legal(atomic_action, player):
                    # logging.error(f"Pick action {atomic_action} is illegal!")
                    return False
            # illegal drop action
            elif atomic_action[0] == 'drop off':
                if not _is_drop_action_legal(atomic_action, player):
                    # logging.error(f"Drop action {atomic_action} is illegal!")
                    return False
            elif atomic_action[0] == 'wait':
                if len(all_passengers) > 0:
                    return False
            else:
                return False
        # check mutex action
        if _is_action_mutex(action):
            # logging.error(f"Actions {action} are mutex!")
            return False
        # check taxis collision
        if len(state['taxis']) > 1:
            taxis_location_dict = dict(
                [(t, state['taxis'][t]['location']) for t in state['taxis'].keys()])
            move_actions = [a for a in action if a[0] == 'move']
            for move_action in move_actions:
                taxis_location_dict[move_action[1]] = move_action[2]
            if len(set(taxis_location_dict.values())) != len(taxis_location_dict):
                # logging.error(f"Actions {action} cause collision!")
                return False
        return True

    def build_actions(self, state):
        actions = {}
        for taxi in self.my_taxis:
            actions[taxi] = set()
            possible_tiles = self.neighbors(state["taxis"][taxi]["location"])
            for tile in possible_tiles:
                actions[taxi].add(("move", taxi, tile))
            if state["taxis"][taxi]["capacity"] > 0:
                for passenger in state["passengers"].keys():
                    if state["passengers"][passenger]["location"] == state["taxis"][taxi]["location"]:
                        actions[taxi].add(("pick up", taxi, passenger))
            for passenger in state["passengers"].keys():
                if (state["passengers"][passenger]["destination"] == state["taxis"][taxi]["location"]
                        and state["passengers"][passenger]["location"] == taxi):
                    actions[taxi].add(("drop off", taxi, passenger))
            if len(state['passengers'].keys()) <= 1:
                actions[taxi].add(("wait", taxi))
        return actions

    def neighbors(self, location):
        """
        Calls the neighbors' Simulator function
        """
        return Simulator.neighbors(self.simulator,location)

class UCTNode:
    def __init__(self, action, parent, children,visit_count, total_score,best_action):
        self.action = action
        self.parent = parent
        self.children = children
        self.visit_count = visit_count
        self.total_score = total_score
        self.best_action = best_action


class DistFunctions:
    def __init__(self, initial):
            self.state = initial
            self.graph = self.build_distances_graph(initial)
            self.shortest_path_distances = self.min_dist(self.graph)
            self.max_d = nx.diameter(self.graph)

    def get_distances(self, graph, node1, node2):
        return graph.get((node1, node2), self.max_d)

    def min_dist(self, graph):
        distances = {}
        for node1 in graph.nodes:
            for node2 in graph.nodes:
                distances[(node1, node2)] = nx.shortest_path_length(graph, node1, node2)
        return distances

    def build_distances_graph(self, initial):
        n, m = len(initial['map']), len(initial['map'][0])
        graph = nx.grid_graph((m, n))
        del_nodes = [node for node in graph if initial['map'][node[0]][node[1]] == 'I']
        graph.remove_nodes_from(del_nodes)
        return graph

