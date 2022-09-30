import tweepy
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain


#To connect to the API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)
#We set ‘wait_on_rate_limit’ and ‘wait_on_rate_limit_notify’ to True.
#There are rate limits when downloading data from Twitter

#Getting all the data from an individual user
me = api.get_user(screen_name = 'KhalidOkewole')#Getting id from myself
id = me.id

#Creating a list of all my followers
user_list = [id]
follower_list = []
for user in user_list:
    followers = []
    try:
        for page in tweepy.Cursor(api.get_follower_ids, user_id=user).pages():
            followers.extend(page)
            print(len(followers))
    except tweepy.TweepyException:
        print("error")
        continue
    follower_list.append(followers)

#Putting followers into DataFrame
df = pd.DataFrame(columns=['source','target']) #Empty DataFrame
df['target'] = follower_list[0] #Set the list of followers as the target column
df['source'] = id #Set my user ID as the source 

#Visualizing network from DataFrame
G = nx.from_pandas_edgelist(df, 'source', 'target') #Turn df into graph
pos = nx.spring_layout(G) #specify layout for visual
print(G.degree[id])


#Plot graph using matPlotLib
f, ax = plt.subplots(figsize=(10, 10))
plt.style.use('ggplot')
#G = G.to_directed()
nodes = nx.draw_networkx_nodes(G, pos, alpha = 0.8)
nodes.set_edgecolor('k')
nx.draw_networkx_labels(G, pos, font_size=8)
nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.2)
#plt.show()


#Getting the followers of my 35 followers
user_list = list(df['target']) #Use the list of followers we extracted in the code above i.e. my 35 followers
for userID in user_list:
    print(userID)
    followers = []
    follower_list = []

    # fetching the user
    user = api.get_user(user_id = userID)

    # fetching the followers_count
    followers_count = user.followers_count

    try:
        for page in tweepy.Cursor(api.get_follower_ids, user_id=userID).pages():
            followers.extend(page)
            print(len(followers))
            if followers_count >= 5000: #Only take first 5000 followers
                break
    except tweepy.TweepyException:
        print("error")
        continue
    follower_list.append(followers)
    temp = pd.DataFrame(columns=['source', 'target'])
    temp['target'] = follower_list[0]
    temp['source'] = userID
    df = df.append(temp)
    df.to_csv("networkOfFollowers.csv")

#Now read the csv and turn the df into a graph using NetworkX.
df = pd.read_csv('networkOfFollowers.csv') #Read into a df
G = nx.from_pandas_edgelist(df, 'source', 'target')#Changes the number of id nodes from 35 -> 29


#plt.show()

#Running some basic network analytics
print("The number of nodes is {}".format(G.number_of_nodes()))

#The following code finds the number of connections each node has i.e. the degree of each node and sorts them in descending order.
#We can also find the most influential nodes in the network using centrality measures.
#The most simple measure of centrality is Degree Centrality
G_sorted = pd.DataFrame(sorted(G.degree, key=lambda x: x[1], reverse=True))
G_sorted.columns = ['user_id','degree']
G_sorted.head()
mb = G_sorted
print("This represents the degree centrality of my whole network in descending order\n {}".format(mb))

print()
print("-------------------------------------")
print()

#Making the number of nodes smaller as there are far too many nodes
G_tmp = nx.k_core(G, 3) #Exclude nodes with degree less than 3
print("The number of nodes is {}".format(G_tmp.number_of_nodes()))

#With our smaller graph we now seperate our graph into smaller groups

partition = community_louvain.best_partition(G_tmp)

#Turn partition into dataframe
partition1 = pd.DataFrame([partition]).T
partition1 = partition1.reset_index()
partition1.columns = ['names','group']

#The following code finds the number of connections each node has i.e. the degree of each node and sorts them in descending order.
#We can also find the most influential nodes in the network using centrality measures.
#The most simple measure of centrality is Degree Centrality
G_sorted = pd.DataFrame(sorted(G_tmp.degree, key=lambda x: x[1], reverse=True))
G_sorted.columns = ['names','degree']
G_sorted.head()
dc = G_sorted
print("This represents the degree centrality in descending order\n {}".format(dc))

#We now have the nodes split into groups and the degree of each node, we then combine these into one DataFrame.
print()
print("--------------------------------------")
print()
combined = pd.merge(dc,partition1, how='left', left_on="names",right_on="names")
print(combined)


#Now we visualize our graph
#4 colours representing each group(0, 1, 2, 3)

#Converting my graph to a directed graph

G_tmp = G_tmp.to_directed()
pos = nx.spring_layout(G_tmp)
f, ax = plt.subplots(figsize=(10, 10))
plt.style.use('ggplot')
#cc = nx.betweenness_centrality(G2)
nodes = nx.draw_networkx_nodes(G_tmp, pos,
                               cmap = plt.cm.Set1,
                               node_color = combined['group'],
                               alpha = 0.8)

nodes.set_edgecolor('black')
#nodes.set_label
nx.draw_networkx_labels(G_tmp, pos, font_size=8)
nx.draw_networkx_edges(G_tmp, pos, width = 1.0, alpha = 0.2, edge_color = 'black')
plt.savefig('twitterFollowers.png')
#plt.show()


#Plotting the degree distribuition
def degree_distribution(G):
    vk = dict(G.degree())
    vk = list(vk.values()) # we get only the degree values
    maxk = np.max(vk)
    mink = np.min(min)
    kvalues= np.arange(0,maxk +1) # possible values of k
    Pk = np.zeros(maxk + 1) # P(k)
    for k in vk:
        Pk[k] = Pk[k] + 1
    Pk = Pk / sum(Pk) # the sum of the elements of P(k) must to be equal to one
    return kvalues,Pk

ks, Pk = degree_distribution(G_tmp)

plt.figure()
plt.loglog(ks,Pk,'bo',10,10)
plt.xlabel("k", fontsize=20)
plt.ylabel("P(k)", fontsize=20)
plt.title("Degree distribution", fontsize=20)
plt.grid(True)
plt.savefig('degree_dist.png') #save the figure into a file
plt.suptitle("In Degree", fontsize = 20, fontweight = 900)
plt.show()


#deg_cent = nx.degree_centrality(T)

#print(nx.info(G, n=id))


#Getting the in-degree
#degrees = [val for (nodes, val) in G.degree()]

 
#print (id)
#me = api.get_user(screen_name = ‘KhalidOkewole’)
#me.id


#Finding out the person with the higest node degree.(Degree Centrality)
#high = api.get_user(user_id = 1170053059155808258)
#print("The screen name of the user is : " + high.screen_name)