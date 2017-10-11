# Social-network-deannonymization
Read report "Network de-annonymization.pdf".


Graph Matching (GM) is a fundamental part of social network reconciliation.


The experiment is the real application of graph match- ing over social networks in order to help advertisement com- panies or sociologist study. As the main algorithm for social graphs matching we have chosen ExpandWhenStuck that is having the best performance over Percolation based solutions for anonymized graphs. We should notice that in many real life graphs, the edges and nodes also contain various attributes, that we could use as additional features with respect to structure of a graph, that depend of an application of graph matching techniques on a particular field. That in our case, nodes in social graphs have the first and second name. Thus we could use this for reducing the search space while candidate pairs generation process. For the resulting experiment we decided to do graph matching over named graph (first name, last name) such information is useful in order to reduce the bottlenecks and it has free access.


Experiment steps.


1) Crawl Kazan city peoples from Vkontakte social network;
2) Crawl Kazan city peoples from Instagram social image
share network;
3) modify the ExpandWhenStuck algorithm for the named
graph and candidate set generation with respect to named
features of vertices;
4) conduct experiment over a little graph of 20000 vertices;
5) conduct over the whole retrieved graphs and reveal the
efficiency;


In order to have automated test of algorithms’s performance over the real graph collected from Instagram and VK in previous section, we should conduct test just over labeled subgraphs with instagram usernames. After retrieving users from VK social graph we got 30000 labeled users, such subgraph consist of 30 components. The Instgram was sliced by the usernames found in VK. As a result we could test the resulting matched pairs by their user names.