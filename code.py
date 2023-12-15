import numpy as np
import networkx as nx
import random
import pandas as pd
import matplotlib.pyplot as plt


BOUGHT = 1
HAVENT_BOUGHT = -1
MID_BOUGHT = 0


def choose_artists():
    ID1 = '212984801'
    ID2 = '316111442'

    x = (int(ID1[-1]) + int(ID2[-1])) % 5
    y = (int(ID1[-2]) + int(ID2[-2])) % 5
    options = [(70, 150), (989, 16326), (144882, 194647), (389445, 390392), (511147, 532992)]
    y = (y + 1) % 5 if x == y else y

    return(*options[x], *options[y])


def ReadData():
    Data0 = pd.read_csv("instaglam0.csv")
    Data1 = pd.read_csv("instaglam_1.csv")
    DataSpotifly = pd.read_csv("spotifly.csv")
    DataToGraph0 = open("instaglam0.csv", "r")
    DataToGraph1 = open("instaglam_1.csv", "r")
    next(DataToGraph0, None)
    next(DataToGraph1, None)
    G0 = nx.parse_edgelist(DataToGraph0, delimiter=',', create_using=nx.Graph(),nodetype=int, data=(('weight', float),))
    G1 = nx.parse_edgelist(DataToGraph1, delimiter=',', create_using=nx.Graph(),nodetype=int, data=(('weight', float),))
    instaglam0Dict = {}
    instaglam1Dict = {}
    spotiflyDict = {}
    spotiflyDictByArtists = {}
    BuildInitDataDict(Data0,instaglam0Dict, "Instaglam Friendship")
    BuildInitDataDict(Data1,instaglam1Dict, "Instaglam Friendship")
    BuildInitDataDict(DataSpotifly, spotiflyDict, "Spotifly plays per user")
    BuildInitDataDict(DataSpotifly,spotiflyDictByArtists,"spotifly per artist")
    return G0,G1,instaglam0Dict,instaglam1Dict, spotiflyDict,spotiflyDictByArtists


def BuildInitDataDict(csvData, dict, typeOfDict): # build dictionaries from data.
    if typeOfDict == "Instaglam Friendship":
        for a,b in zip(list(csvData["userID"]), list(csvData["friendID"])):
            if a not in dict.keys():
                dict[a] = [b]
            else:
                dict[a].append(b)
            if b not in dict.keys():
                dict[b] = [a]
            else:
                dict[b].append(a)
    else:
        relevant_artists = choose_artists()
        if typeOfDict == "Spotifly plays per user":
            for a, b, c in zip(list(csvData["userID"]), list(csvData[" artistID"]), list(csvData["#plays"])):
                if b in relevant_artists:
                    if a not in dict.keys():
                        dict[a] = [(b, c)]
                    else:
                        dict[a].append((b, c))
                else:
                    if a not in dict.keys():
                        dict[a] = [(b, 0)]
                    else:
                        dict[a].append((b, 0))
        else:
            for a, b, c in zip(list(csvData["userID"]), list(csvData[" artistID"]), list(csvData["#plays"])):
                if b in relevant_artists:
                    if b not in dict.keys():
                        dict[b] = [a]
                    else:
                        dict[b].append(a)


def add_attributes(graph, artists): # Add an infection status attribute to all nodes.
    for artist in artists:
        nx.set_node_attributes(graph, HAVENT_BOUGHT, "color" + str(artist))
# ------------------------------------------------------------------------------------------------------------------
# Probability of edge creation calculations functions:


def RA_improvedN(G, i, j, max, rank):
    return RA_improved_score(G, i, j, rank)/max


def findRA_improved_max(G, rank):
    max = 0
    for i, node1 in enumerate(G.nodes()):
        for j, node2 in enumerate(G.nodes()):
            if i < j and not G.has_edge(node1, node2):
                score = RA_improved_score(G, node1, node2, rank)
                if max < score:
                    max = score
    return max


def RA_improved_score(G, i, j, rank):
    neighbours = [z for z in list((set(G.adj[i]).intersection(set(G.adj[j]))))]
    N_T_plus = []
    N_T_minus = []
    for z in neighbours:
        rankZ = len(G.adj[z])
        if rankZ < rank:
            N_T_plus.append(rankZ)
        else:
            N_T_minus.append(rankZ)
    return sum([1/item for item in N_T_minus]) + sum([1/(item**2) for item in N_T_plus])
# ------------------------------------------------------------------------------------------------------------------


def add_edges(graph, prob_func, max_measure): # Add edges to graph.
    edges = []
    for i, node1 in enumerate(graph.nodes()):
        for j, node2 in enumerate(graph.nodes()):
            if i < j and not graph.has_edge(node1, node2):
                p = random.random()
                prob = prob_func(node1, node2, graph, max_measure)
                if p <= prob:
                    edges.append((node1, node2))
    graph.add_edges_from(edges)


def get_influencers(graph, artists, spotifly_dict): # Find influencers according to pagerank and an influence measure.
    alpha = 0.5
    influencers_per_artist = {}
    node2page = nx.algorithms.link_analysis.pagerank(graph, alpha=0.95, max_iter=100, weight="weight")

    for artist in artists:
        top_influencers = {}
        potential_influencers = list(calc_influence(graph, artist, spotifly_dict, alpha).keys())[:100]
        for influencer in potential_influencers:
            top_influencers[influencer] = node2page[influencer]
        influencers_per_artist[artist] = sorted(top_influencers.items(), key=lambda item: item[1], reverse=True)[:5]
        influencers_per_artist[artist] = [t[0] for t in influencers_per_artist[artist]]

    return influencers_per_artist


def calc_influence(graph, artist, spotifly_dict, alpha): # calculate influence per node.
    influence_per_node = {}
    for node in graph.nodes():
        neighbors_num = len(graph.adj[node])
        plays_num = 0
        for neighbor in graph.adj[node]:
            for tuple in spotifly_dict[neighbor]:
                if artist == tuple[0]:
                    plays_num += tuple[1]
        influence_per_node[node] = alpha*neighbors_num + (1-alpha)*plays_num/(neighbors_num + 1)
    nodes_influence = {key: value for key, value in sorted(influence_per_node.items(),
                                                           key=lambda item: item[1], reverse=True)}
    return nodes_influence


def pick_influencers(graph, influencers): # infect influencers.
    for key, value in influencers.items():
        for influencer in value:
            graph.nodes[influencer]["color" + str(key)] = BOUGHT


def check_who_bought(graph, artists, spotifly_dict,spotiflyByArtists): # Infection process.
    new_buyers_per_artist = {}
    for artist in artists:
        new_buyers_per_artist[artist] = 0
    for node in graph.nodes():
        for artist in artists:
            Bt = 0
            if graph.nodes[node]["color" + str(artist)] == HAVENT_BOUGHT:
                for neighbor in graph.adj[node]:
                    if graph.nodes[neighbor]["color" + str(artist)] == BOUGHT:
                        Bt += 1
                Nt = len(graph.adj[node])
                p = random.uniform(0, 1)
                if node in spotiflyByArtists[artist]:
                    for tuple in spotifly_dict[node]:
                        if artist == tuple[0]:
                            h = tuple[1]
                            if p <= (h * Bt) / (1000 * Nt):
                                graph.nodes[node]["color" + str(artist)] = MID_BOUGHT
                else:
                    if p <= Bt / Nt:
                        graph.nodes[node]["color" + str(artist)] = MID_BOUGHT

    for node in graph.nodes():
        for artist in artists:
            if graph.nodes[node]["color" + str(artist)] == MID_BOUGHT:
                graph.nodes[node]["color" + str(artist)] = BOUGHT
                new_buyers_per_artist[artist] += 1

    return graph, new_buyers_per_artist


def simulation_process(graph, artists, spotifly_dict, prob_func, spotiflydict_byartist):
    total_sales_per_artist = {}

    for artist in artists:
        total_sales_per_artist[artist] = 0
    for t in range(1, 7):
        max_measure = findRA_improved_max(graph, 10)
        add_edges(graph, prob_func, max_measure)
        graph, buyers_at_t_dict = check_who_bought(graph, artists, spotifly_dict, spotiflydict_byartist)
        for artist in artists:
            total_sales_per_artist[artist] += buyers_at_t_dict[artist]

    return graph, total_sales_per_artist


def main():
    G0, G1, instaglam0Dict, instaglam1Dict, spotiflydict, spotiflydict_byartist = ReadData()
    artists = choose_artists()
    add_attributes(G0, artists)
    ProbFunc = lambda i, j, G, maxRA_improved: RA_improvedN(G, i, j, maxRA_improved, 10) * 0.19

    # Simulation with final influencers.
    influencers = {}
    influencers[389445] = [548221, 874459, 411093, 999659, 983003]
    influencers[390392] = [548221, 874459, 411093, 441435, 175764]
    influencers[511147] = [548221, 874459, 411093, 441435, 999659]
    influencers[532992] = [548221, 874459, 411093, 441435, 999659]

    pick_influencers(G0, influencers)

    G_final, total_sales = simulation_process(G0, artists, spotiflydict, ProbFunc, spotiflydict_byartist)
    nodes_num = len(G_final.nodes())
    sales_percentage = []

    for artist in artists:
        sales_percentage.append("Artist " + f'{artist}: {(total_sales[artist] / nodes_num) * 100}')

    print(sales_percentage)



    """Not in use functions - HW process"""

    """**************************************************************************************************************"""

    """Part A: initial Data gathering"""

    """**************************************************************************************************************"""

    def findNewEdges(G0,G1):
        newEdges = []
        for edge in G0.edges:
            if edge not in G1.edges:
                if edge[0] < edge[1]:
                    newEdges.append((edge[0],edge[1]))
                else:
                    newEdges.append((edge[1],edge[0]))
        return newEdges

    def findPercentage(networkNow,networkBefore, numCreated):
        sum = 0
        for i, node1 in enumerate(networkBefore.nodes()):
            for j, node2 in enumerate(networkBefore.nodes()):
                if i < j and not networkBefore.has_edge(node1, node2) and not networkNow.has_edge(node1, node2):
                    sum += 1
        print("percentage of edges created:")
        print(numCreated/sum)

    def dataOnConnections(newEdges, dictBefore, networkBefore, networkNow):
        sharedFriendDict = {}
        commonConnectorsDict = {}
        friendsDict = {}
        list_shared = []
        list_noE_shared = []
        list_friends = []
        list_noE_friends = []
        for edge in newEdges:
            sharedFriends = list(set(dictBefore[edge[0]]).intersection(dictBefore[edge[1]]))
            numShared = len(sharedFriends)
            addToConnectors(commonConnectorsDict,sharedFriends,dictBefore)
            addToHist(sharedFriendDict,numShared)
            addToHistFriends(friendsDict, len(dictBefore[edge[0]]), len(dictBefore[edge[1]]))

            list_friends.append(len(dictBefore[edge[0]]))
            list_friends.append(len(dictBefore[edge[1]]))
            list_shared.append(numShared)

        for i, node1 in enumerate(networkBefore.nodes()):
            for j, node2 in enumerate(networkBefore.nodes()):
                if i < j and not networkBefore.has_edge(node1, node2) and not networkNow.has_edge(node1, node2):
                    list_noE_friends.append(len(dictBefore[node1]))
                    list_noE_friends.append(len(dictBefore[node2]))
                    list_noE_shared.append(len(list(set(dictBefore[node1]).intersection(dictBefore[node2]))))

        # Plotting all Data explored
        plt.hist(list_shared, bins=19, ec="black")
        plt.xticks(np.arange(0, 22, 1))
        plt.title("number of shared friends between 2 newly connected nodes")
        plt.xlabel("number of shared friends")
        plt.ylabel("number of new edges")
        plt.show()

        plt.hist(list_noE_shared, bins=200, ec="black")
        plt.title("number of shared friends between 2 nodes that could have been connected")
        plt.xlabel("number of shared friends")
        plt.ylabel("number of new edges")
        plt.show()

        plt.hist(list_friends, bins=25, ec="black")
        plt.title("number of friends of nodes with new connection")
        plt.xlabel("number of friends")
        plt.ylabel("number of new edges")
        plt.show()

        plt.hist(list_noE_friends, bins=80, ec="black")
        plt.title("number of friends of nodes that an edge didnt formed between them")
        plt.xlabel("number of friends")
        plt.ylabel("number of new edges")
        plt.show()

        return {k: v for k, v in sorted(sharedFriendDict.items(), key=lambda item: item[1])}, \
               {k: v for k, v in sorted(commonConnectorsDict.items(), key=lambda item: item[1])}, \
               {k: v for k, v in sorted(friendsDict.items(), key=lambda item: item[1])}


    def addToConnectors(dict,sharedFriends,dictBefore):
        for friend in sharedFriends:
            if (friend,len(dictBefore[friend])) not in dict.keys():
                dict[(friend,len(dictBefore[friend]))] = 1
            else:
                dict[(friend,len(dictBefore[friend]))] += 1


    def addToHist(sharedFriendDict,numShared):
        if numShared not in sharedFriendDict.keys():
            sharedFriendDict[numShared] = 1
        else:
            sharedFriendDict[numShared] += 1

    def addToHistFriends(friendsDict, numberFriends1, numberFriends2):
        if numberFriends1 not in friendsDict.keys():
            friendsDict[numberFriends1] = 1
        else:
            friendsDict[numberFriends1] += 1

        if numberFriends2 not in friendsDict.keys():
            friendsDict[numberFriends2] = 1
        else:
            friendsDict[numberFriends2] += 1


    def BuildNormalizedDict(network0, networkBefore, histDict):
        dict_mutual_neighbors = {}
        for i, node1 in enumerate(networkBefore.nodes()):
            for j, node2 in enumerate(networkBefore.nodes()):
                if i < j and not networkBefore.has_edge(node1, node2):
                    mutual_neighbors = len(set(networkBefore.adj[node1]).intersection(set(networkBefore.adj[node2])))
                    if mutual_neighbors not in dict_mutual_neighbors.keys() and mutual_neighbors in histDict.keys():
                        dict_mutual_neighbors[mutual_neighbors] = [0, 0]
                    if mutual_neighbors in histDict.keys():
                        dict_mutual_neighbors[mutual_neighbors][1] += 1
                        if not networkBefore.has_edge(node1, node2) and network0.has_edge(node1, node2):
                            dict_mutual_neighbors[mutual_neighbors][0] += 1

        for key, value in dict_mutual_neighbors.items():
            dict_mutual_neighbors[key] = value[0]/value[1]

        # Plotting
        plt.scatter(list(dict_mutual_neighbors.keys()), list(dict_mutual_neighbors.values()))
        plt.title("normalized frequencies of shared friends")
        plt.xlabel("number of shared friends")
        plt.xticks(np.arange(0, 25, 1))
        plt.ylabel("normalized frequencies")
        plt.show()

        return dict_mutual_neighbors

    """**************************************************************************************************************"""

    """Part B: Find probability function """

    """**************************************************************************************************************"""

    """: Logistic Regression"""

    # Logistic Regression Class
    class LogisticRegression:
        def __init__(self,x,y):
            self.intercept = np.ones((x.shape[0], 1))
            self.x = np.concatenate((self.intercept, x), axis=1)
            self.weight = np.zeros((self.x.shape[1],1))
            self.y = y

        #Sigmoid method
        def sigmoid(self, x, weight):
            z = np.dot(x, weight)
            return 1 / (1 + np.exp(-z))

        #method to calculate the Loss
        def loss(self, h, y):
            return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

        #Method for calculating the gradients
        def gradient_descent(self, X, h, y):
            return np.dot(X.T, (h - y)) / y.shape[0]


        def fit(self, lr , iterations):
            for i in range(iterations):
                sigma = self.sigmoid(self.x, self.weight)
                sigma = sigma.reshape((sigma.shape[0],1))

                #loss = self.loss(sigma,self.y)

                dW = self.gradient_descent(self.x , sigma, self.y)

                #Updating the weights
                self.weight -= lr * dW

            return print('fitted successfully to data')

        #Method to predict the class label.
        def predict(self, x_new ):
            x_new = np.concatenate((self.intercept, x_new), axis=1)
            result = self.sigmoid(x_new, self.weight)
            return result

    def buildLabelsAndFeatures(G_before, G_now):
        numCurrentNodes = len(G_before.nodes())
        size = int((numCurrentNodes*(numCurrentNodes-1))/2 - len(G_before.edges()))
        dataMatrix = np.zeros((size,2)) # matrix of all edges
        iter = 0
        y = np.zeros((size,1)) # labels of all not created yet edges
        dictOrders = {}
        edges = []
        for i, firstNode in enumerate(G_before.nodes()):
            for j, secondNode in enumerate(G_before.nodes()):
                if i < j and not G_before.has_edge(firstNode,secondNode):
                    edges.append((firstNode,secondNode))
                    sharedFriends = len(set(G_before.adj[firstNode]).intersection(set(G_before.adj[secondNode])))
                    allFriends = len(set(G_before.adj[firstNode]).union(set(G_before.adj[secondNode])))
                    sharedFriendsList = set(G_before.adj[firstNode]).intersection(set(G_before.adj[secondNode]))
                    adamicScore = 0
                    for node in sharedFriendsList:
                        adamicScore += (1/(np.log(len(G_before.adj[node]))))
                    jaccardScore = sharedFriends/allFriends
                    dataMatrix[iter][0] = sharedFriends
                    dataMatrix[iter][1] = adamicScore
                    if firstNode < secondNode:
                        dictOrders[(firstNode,secondNode)] = iter
                    else:
                        dictOrders[(secondNode,firstNode)] = iter
                    if G_now.has_edge(firstNode,secondNode):
                        y[iter] = 1
                    else:
                        y[iter] = 0
                    iter+=1

        return y, dataMatrix

    def runLogiReg(newtorkBefore, networkNow):
        dataMatrix, labels = buildLabelsAndFeatures(newtorkBefore,networkNow)
        regressor = LogisticRegression(dataMatrix,labels)
        regressor.fit(0.01,1000)
        y_pred = regressor.predict(dataMatrix)

    """Final Prob simulation code"""

    def simulateEdges(networkNow,findProb):
        count = 0
        dictProbs = {}
        probs = []
        maxRA_improved = findRA_improved_max(networkNow,10)
        for i, node1 in enumerate(networkNow.nodes()):
            for j, node2 in enumerate(networkNow.nodes()):
                if i < j and not networkNow.has_edge(node1, node2):
                    t = random.random()
                    probCurrentEdge = findProb(node1,node2,networkNow, maxRA_improved)
                    probs.append(probCurrentEdge)
                    if t <= probCurrentEdge:
                        count += 1
                        if node1 < node2:
                            dictProbs[(node1,node2)] = probCurrentEdge
                        else:
                            dictProbs[(node2,node1)] = probCurrentEdge

        return count, {key: value for key, value in sorted(dictProbs.items(), key=lambda item: item[1], reverse=True)}

    def simulation(ProbFunc, newtorkNow, newEdges):
        sum = 0
        for i in range(5):
            count, probDict = simulateEdges(newtorkNow,ProbFunc)
            if i == 0:
                print("*******************************************************")
            print("created: " + str(len(probDict)))
            intersection = len(set(probDict.keys()).intersection(set(newEdges)))
            print("wanted edges in common: " +  str(intersection))
            print("edge with highest Prob: ", list(probDict.items())[0], " | edge with smallest Prob: ", list(probDict.items())[len(probDict)-1] )
            print("*******************************************************")
            sum += intersection

        print("result: " + str((sum/5)/1607))

    """**************************************************************************************************************"""

    """Part C: influencers"""

    """**************************************************************************************************************"""

    """first few attempts to find influencers"""

    #All permutation of size n from given set
    def permutations(iterable, n):
        tup = tuple(iterable)
        k = len(tup)
        if n > k:
            return
        indices = list(range(n))
        yield list(tup[i] for i in indices)
        while True:
            for i in reversed(range(n)):
                if indices[i] != i + k - n:
                    break
            else:
                return
            indices[i] += 1
            for j in range(i + 1, n):
                indices[j] = indices[j - 1] + 1
            yield list(tup[i] for i in indices)


    # First attempt :
    # Calculate for each node its influence measure by node degree and sum neighbors plays for each artist
    def influence_measure(network, artist, alpha):
        nodes_measures = {}
        for node in network.nodes():
            count_neighbors = 0
            count_plays = 0
            for neighbor in network.adj[node]:
                if artist in network.nodes[neighbor]["artists"].keys():
                    count_neighbors += 1
                    count_plays += network.nodes[neighbor]["artists"][artist]
            nodes_measures[node] = alpha*count_neighbors + (1-alpha)*(count_plays)
        nodes_measures = {key: value for key, value in sorted(nodes_measures.items(), key=lambda item: item[1], reverse=True)}
        return nodes_measures


    # From nodes with large measure as written above taking 5 highest page ranks
    def choose_influencers(network, artists):
        artistsInfluencers = {}
        alpha = 0.5
        node2page = nx.algorithms.link_analysis.pagerank(network, max_iter=100, weight="weight")

        for artist in artists:
            influencers = {}
            potential_influencers = list(influence_measure(network, artist, alpha).keys())[:100]
            for node in potential_influencers:
                influencers[node] = node2page[node]
            artistsInfluencers[artist] = sorted(influencers.items(), key=lambda item: item[1], reverse=True)[:10]
            artistsInfluencers[artist] = [t[0] for t in artistsInfluencers[artist]]

        # Second attempt:
        # Checking distances between potential influencers
        # first priority: inf distance (unreachable nodes), second priority: sum of distances
        dist_matrix = nx.floyd_warshall_numpy(network, weight='weight')
        for artist in artists:
            permutationsList = list(permutations(artistsInfluencers[artist], 5))
            max_inf = 0
            max_inf_group = []
            groups_dict = {}
            for group in permutationsList:
                sum_dists = 0
                sum_inf = 0
                for node1 in group:
                    for node2 in group:
                        index_1 = list(network.nodes()).index(node1)
                        index_2 = list(network.nodes()).index(node2)
                        if str(dist_matrix[index_1][index_2]) != "inf":
                            sum_dists += dist_matrix[index_1][index_2]
                        else:
                            sum_inf += 1
                if sum_inf == max_inf:
                    max_inf_group.append(group)
                if sum_inf > max_inf:
                    max_inf = sum_inf
                    max_inf_group = [group]
                groups_dict[group] = sum_dists
            if len(max_inf_group) == 1:
                artistsInfluencers[artist] = max_inf_group[0]
            if len(max_inf_group) == 0:
                max_inf_group = permutations
            max_dists = 0
            max_group = max_inf_group[0]
            for group in max_inf_group:
                if groups_dict[group] > max_dists:
                    max_dists = groups_dict[group]
                    max_group = group
            artistsInfluencers[artist] = max_group

        return artistsInfluencers


    # From potential influencers check those who have the largest amount of
    # "good" neighbors minus neighbors with no plays or over 1000 plays
    def find_good_neighbors(potential, network, artists):
        goodNeighbors = {}
        for artist in artists:
            goodNeighbors[artist] = {}
        for artist in artists:
            for node in potential:
                count = 0
                for neighbor in network.adj[node]:
                    if (artist not in network.nodes[neighbor]["artists"].keys()) \
                            or (artist in network.nodes[neighbor]["artists"].keys() and network.nodes[neighbor]["artists"][artist] > 1000):
                        count += 1
                goodNeighbors[artist][node] = count
        for artist in artists:
            goodNeighbors[artist] = {key: value for key, value in sorted(goodNeighbors[artist].items(), key=lambda item: item[1], reverse=True)}
            goodNeighbors[artist] = list(goodNeighbors[artist].keys())[:5]

        for key, value in goodNeighbors.items():
            print(f"{key}: {value}")

        return goodNeighbors


    """Check infection percentage per permutation."""

    print([548221, 874459, 411093, 441435, 175764])
    for artist in artists:
        influencers[artist] = [548221, 874459, 411093, 441435, 175764]
    pick_influencers(G0, influencers)
    G_final, total_sales = simulation_process(G0, artists, spotiflydict, ProbFunc, spotiflydict_byartist)
    nodes_num = len(G_final.nodes())
    sales_percentage = []

    for artist in artists:
        sales_percentage.append("Artist " + f'{artist}: {(total_sales[artist] / nodes_num) * 100}')

    print(sales_percentage)

    print([548221, 874459, 411093, 441435, 999659])
    for artist in artists:
        influencers[artist] = [548221, 874459, 411093, 441435, 999659]
    pick_influencers(G0, influencers)
    G_final, total_sales = simulation_process(G0, artists, spotiflydict, ProbFunc, spotiflydict_byartist)
    nodes_num = len(G_final.nodes())
    sales_percentage = []

    for artist in artists:
        sales_percentage.append("Artist " + f'{artist}: {(total_sales[artist] / nodes_num) * 100}')

    print(sales_percentage)

    print([548221, 874459, 411093, 441435, 983003])
    for artist in artists:
        influencers[artist] = [548221, 874459, 411093, 441435, 983003]
    pick_influencers(G0, influencers)
    G_final, total_sales = simulation_process(G0, artists, spotiflydict, ProbFunc, spotiflydict_byartist)
    nodes_num = len(G_final.nodes())
    sales_percentage = []

    for artist in artists:
        sales_percentage.append("Artist " + f'{artist}: {(total_sales[artist] / nodes_num) * 100}')

    print(sales_percentage)

    print([548221, 874459, 411093, 175764, 999659])
    for artist in artists:
        influencers[artist] = [548221, 874459, 411093, 175764, 999659]
    pick_influencers(G0, influencers)
    G_final, total_sales = simulation_process(G0, artists, spotiflydict, ProbFunc, spotiflydict_byartist)
    nodes_num = len(G_final.nodes())
    sales_percentage = []

    for artist in artists:
        sales_percentage.append("Artist " + f'{artist}: {(total_sales[artist] / nodes_num) * 100}')

    print(sales_percentage)

    print([548221, 874459, 411093, 175764, 983003])
    for artist in artists:
        influencers[artist] = [548221, 874459, 411093, 175764, 983003]
    pick_influencers(G0, influencers)
    G_final, total_sales = simulation_process(G0, artists, spotiflydict, ProbFunc, spotiflydict_byartist)
    nodes_num = len(G_final.nodes())
    sales_percentage = []

    for artist in artists:
        sales_percentage.append("Artist " + f'{artist}: {(total_sales[artist] / nodes_num) * 100}')

    print(sales_percentage)

    print([548221, 874459, 411093, 999659, 983003])
    for artist in artists:
        influencers[artist] = [548221, 874459, 411093, 999659, 983003]
    pick_influencers(G0, influencers)
    G_final, total_sales = simulation_process(G0, artists, spotiflydict, ProbFunc, spotiflydict_byartist)
    nodes_num = len(G_final.nodes())
    sales_percentage = []

    for artist in artists:
        sales_percentage.append("Artist " + f'{artist}: {(total_sales[artist] / nodes_num) * 100}')

    print(sales_percentage)

    print([548221, 874459, 441435, 175764, 999659])
    for artist in artists:
        influencers[artist] = [548221, 874459, 441435, 175764, 999659]
    pick_influencers(G0, influencers)
    G_final, total_sales = simulation_process(G0, artists, spotiflydict, ProbFunc, spotiflydict_byartist)
    nodes_num = len(G_final.nodes())
    sales_percentage = []

    for artist in artists:
        sales_percentage.append("Artist " + f'{artist}: {(total_sales[artist] / nodes_num) * 100}')

    print(sales_percentage)

    print([548221, 874459, 441435, 175764, 983003])
    for artist in artists:
        influencers[artist] = [548221, 874459, 441435, 175764, 983003]
    pick_influencers(G0, influencers)
    G_final, total_sales = simulation_process(G0, artists, spotiflydict, ProbFunc, spotiflydict_byartist)
    nodes_num = len(G_final.nodes())
    sales_percentage = []

    for artist in artists:
        sales_percentage.append("Artist " + f'{artist}: {(total_sales[artist] / nodes_num) * 100}')

    print(sales_percentage)

    print([548221, 874459, 441435, 999659, 983003])
    for artist in artists:
        influencers[artist] = [548221, 874459, 441435, 999659, 983003]
    pick_influencers(G0, influencers)
    G_final, total_sales = simulation_process(G0, artists, spotiflydict, ProbFunc, spotiflydict_byartist)
    nodes_num = len(G_final.nodes())
    sales_percentage = []

    for artist in artists:
        sales_percentage.append("Artist " + f'{artist}: {(total_sales[artist] / nodes_num) * 100}')

    print(sales_percentage)

    print([548221, 874459, 175764, 999659, 983003])
    for artist in artists:
        influencers[artist] = [548221, 874459, 175764, 999659, 983003]
    pick_influencers(G0, influencers)
    G_final, total_sales = simulation_process(G0, artists, spotiflydict, ProbFunc, spotiflydict_byartist)
    nodes_num = len(G_final.nodes())
    sales_percentage = []

    for artist in artists:
        sales_percentage.append("Artist " + f'{artist}: {(total_sales[artist] / nodes_num) * 100}')

    print(sales_percentage)

    """Graph and nodes characterization: Diameter, Connected components, centrality measures, etc"""

    fans_per_component = {}
    fans_num = 0
    for artist in artists:
        fans_per_component[artist] = []
        for component in nx.connected_components(G0):
            for node in component:
                for tuple in spotiflydict[node]:
                    if tuple[0] == artist:
                        fans_num += 1
            fans_per_component[artist].append(fans_num)
            fans_num = 0
    for item in fans_per_component.items():
        print(f"{item[0]}: {item[1]}")
    print([len(c) for c in nx.connected_components(G0)])

    node_degree_dict = G0.degree()
    print(sorted(node_degree_dict, key=lambda item: item[1], reverse=True)[:50])
    top_five = list((x[0] for x in sorted(node_degree_dict, key=lambda item: item[1], reverse=True)[:5]))
    print(top_five)

    nodes_degree_centrality = nx.degree_centrality(G0)
    nodes_betwenness = nx.betweenness_centrality(G0)
    nodes_eigenvector = nx.eigenvector_centrality(G0, 100)
    nodes_closeness = nx.closeness_centrality(G0)
    nodes_dispersion = sorted(nx.dispersion(G0, 548221).items(), key=lambda t: t[1], reverse=True)
    nodes_load = nx.load_centrality(G0)
    nodes_harmonic = nx.harmonic_centrality(G0)
    print(sorted(nodes_betwenness.items(), key=lambda item: item[1], reverse=True)[:50])
    print(sorted(nodes_closeness.items(), key=lambda item: item[1], reverse=True)[:50])
    print(sorted(nodes_degree_centrality.items(), key=lambda item: item[1], reverse=True)[:50])
    print(sorted(nodes_harmonic.items(), key=lambda item: item[1], reverse=True)[:50])
    print(sorted(nodes_eigenvector.items(), key=lambda item: item[1], reverse=True)[:50])
    print(sorted(nodes_load.items(), key=lambda item: item[1], reverse=True)[:50])

    print(sorted(nx.shortest_path_length(G0, 548221).items(), key=lambda item: item[1], reverse=True))
    print(sorted(nx.shortest_path_length(G0, 469062).items(), key=lambda item: item[1], reverse=True))
    print(sorted(nx.shortest_path_length(G0, 90287).items(), key=lambda item: item[1], reverse=True))


    print(sorted(nx.shortest_path_length(G0, 548221).items(), key=lambda t: t[1], reverse=True))
    print(set(nx.articulation_points(G0)).intersection(set(c[0] for c in sorted(nx.betweenness_centrality(G0).items(), key=lambda t: t[1], reverse=True)[:10])))
    print(sorted(nx.shortest_path_length(G0, 873017).items(), key=lambda t: t[1], reverse=True))
    print(nodes_betwenness[873017])
    print(nodes_betwenness[469062])

    nx.voterank(G0, 50)
    print(nx.algorithms.community.greedy_modularity_communities(G0))
    k = nx.average_neighbor_degree(G0)
    for node in nodes_betwenness:
        print(node[0], k[node[0]])
    print(sorted(nodes_betwenness.items(), key=lambda item: item[1], reverse=True))
    print(list(nx.articulation_points(G0)))

    """**************************************************************************************************************"""


if __name__ == "__main__":
    main()
