#This folder represents the implementation of the embedding algorithm Walklets, for graph encoding
**<span style="color:red">"Don't Walk, Skip! Online Learning of Multi-scale Network Embeddings"**</span>
<https://arxiv.org/abs/1605.02115>

Project structure:
- alg_base_code-->random_walk.py=implementation of the base Random Walk algorithm(corpus obtained via extracting k=1 distanced pairs) used in RandomWalk  
	       -->walklets.py=implemenation of the Walklets algorithm, using at base the corpus obtained via Random Walk, with k-scaled pairs  
- alg_base_explained_notebook=notebook that explains the impl of alg_base_code and other methods used in testing, along with the problems that were encountered, and how were managed
- datasets=includes the Cora dataset used for training

- demo=an Demo App, showing the capabilities of the Walklets algorithm, mini-implementation of the walklets_test.ipynb file
- docs=containing the pptx
- models_emb=embeddings of the nodes of the graphs used for training
- tests=notebook containing the test file of the algorithm
- utils=includes the methods used in visualisation of the test results, and for using pygraphs and and artifical made of the Word2Vec model(gensim not mantained with the current numpy version)

*Credits:Edward326*
