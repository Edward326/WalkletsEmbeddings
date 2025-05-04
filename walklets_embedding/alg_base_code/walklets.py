"""
This module implements the Walklets algorithm for network embedding
Paper used for implementation:
"Don't Walk, Skip! Online Learning of Multi-scale Network Embeddings"
<https://arxiv.org/abs/1605.02115>

This algorithm uses at it its basis random walks algorithm
but walklets captures far more details, in e.g than DeepWalk, algorithm
where only random walks are used.Through walklets we can preserve the multiscale
property of the graph, meaning that we can use the embeddings generated,
in order to help us, differentiate the communities
that are very closed together,to learn more about the structures that are very tied together.

Walklets produce embeddings that are known here as latent multi-scale representation.
They are named multi-scale, because the algorithm let us choose a scale k at which
the embeddings will be obtained.
By modifying scale k, we can control what communities are preserved in a node, higher or lower value of k,
will capture larger or smaller communities, a node will store infos for its far away or closer neighbours.

As every embedding algorithm differs only at the format of the dataset, that will be used for training,
instead of using the classical walks generated from random walks as train dataset, we will extract nodes from
that walks that are k distance apart(links between that nodes, could be found in the adjency matrix power at k),
and form pairs with that nodes, and put in the dataset.Then the dataset will be fed to a Embedding Model, that will
learn the represenation of the nodes, here we will use the Word2Vec model.

The Walklets algorithm is presented in the class below.
When creating an instance, you will need 
to input several args through the constructor in order to set the paramaters for the walks that will be generated,and for Word2Vec Model.
First the algorithm will generate random walks from the original graph, then it will extract pairs of nodes that are k distance apart,
than give it to the Word2Vec model, and train the embeddings.
After the training, you could get the embeddings for all the nodes through the get_embedding method,or 
get the embeddings for certain nodes, through the get_embedding_specific method.

**The Walklets will perform an embedding on all the nodes, meaning it will construct walks from all nodes, than
if you want to get the embeddings for a specific node, you will need to use the get_embedding_specific method(in e.g
if you want the embbedings of nodes from training set)*
"""
import numpy as np
import networkx as nx
import os
import json
from typing import List, Any, Union, Tuple, Dict
from .random_walks import RandomWalkEmbedder
# Import custom Word2Vec implementation
import os
import sys
parent_dir = os.path.abspath(os.path.join(".."))
if not parent_dir in sys.path:
    sys.path.append(parent_dir)
from utils.word2vec import Word2Vec

class WalkletsEmbedder:
    def __init__(
        self,
        seed: int = 42,
        walk_number: int = 1000,
        walk_length: int = 11,
        embed_dim: int = 128,
        window_size: int = 3,
        workers: int = 4,
        epochs: int = 250,#enough for training time
        learning_rate: float = 0.025,
    ):
        """
        The method initializes the Random Walk algorithm.
        Returns:
            nothing
        Args:
            seed (int): Random seed value. Default is 42.
            walk_number (int): Number of random walks. Default is 1000.
            walk_length (int): Length of random walks. Default is 11.
            embed_dim (int): Dimensionality of embedding. Default is 128.
            window_size (int): Size of the context window. Default is 3
            e.g. walk->[0,1,2,3,4,5] and window_size=3 for node 4 word2vec will use as target: (4,1),(4,2),(4,3),(4,5)
            workers (int): Number of cores to be used by the Word2Vec Model. Default is 4.
            epochs (int): Number of epochs. Default is 500.
            learning_rate (float): Learning rate of the Word2Vec Model. Default is 0.025.
        """
        self.walk_number = walk_number
        self.walk_length = walk_length
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.workers = workers
        self.seed = seed
        #self.embeddings = None
        self.scale_embeddings = {}  # Dictionary to store embeddings for each scale
        self.scales = []  # List to keep track of used scales
        self.node_to_idx = None
        np.random.seed(self.seed)  # Set the random seed
    
    @staticmethod
    def _select_walklets(walks: List[List[str]], scale: int=2) -> List[List[str]]:
        """
        Extracts nodes that are separated by 'scale' steps in each walk to form a new corpus.
        Args:
            walks (List[List[str]]): The list of walks where each walk is a list of node IDs as strings
            scale (int): The scale k at which to extract nodes   
        Returns:
            List[List[str]]: A list of walklet sequences with nodes that are scale steps apart
        """
        walklets = []
        for walk in walks:
            if len(walk) <= scale:
                continue
            
            # Create a new walk with nodes that are scale steps apart
            walklet = [walk[i] for i in range(0, len(walk), scale)]
            if len(walklet) > 1:  # Only add walks with at least 2 nodes
                walklets.append(walklet)
        
        return walklets
        
    def fit(self, original_graph: nx.classes.graph.Graph = nx.karate_club_graph(), scales: Union[List[int], int] = 1):
        """
        The method fits the Walklets model to the graph.
        It will create random walks on the graph provided, then for each scale it will
        extract nodes that are k distance apart from these walks, creating separate corpus for each scale.
        Then it will train a Word2Vec model for each scale and concatenate all embeddings.
        
        Returns:
            nothing
        Args:
            original_graph (nx.Graph): The original graph to be used for embedding. Default is the karate club graph.
            scales (Union[List[int], int]): The scale(s) of the walklets. Can be a single integer or a list of integers.
                                           Default is 1 (equivalent to DeepWalk).
        """
        # Convert single scale to list for consistent processing
        if isinstance(scales, int):
            scales = [scales]
        
        # Save scales used
        self.scales = scales
        
        # Initialize node to index mapping
        self.node_to_idx = {str(node): idx for idx, node in enumerate(original_graph.nodes())}
        num_of_nodes = original_graph.number_of_nodes()
        
        # Generate random walks using the static method from RandomWalkEmbedder
        # We generate the walks once and reuse them for different scales
        walks = RandomWalkEmbedder.walk_graph(original_graph, self.walk_number, self.walk_length)
        print(f"Generated a total of {len(walks)} random walks of length {self.walk_length}, starting from {len(walks)/self.walk_number} nodes.\n")
        
        # Initialize embeddings array
        all_embeddings = []
        
        # Process each scale
        for scale in scales:
            print(f"Processing scale k={scale}")
            
            # Extract node sequences that are 'scale' steps apart
            scale_corpus = self._select_walklets(walks, scale)
            print(f"Extracted {len(scale_corpus)} walklets for scale k={scale}")
            
            # Train custom Word2Vec model
            model = Word2Vec(
                sequences=scale_corpus,
                embedding_dim=self.embed_dim,
                window_size=self.window_size,  # Using standard window size for the new corpus
                negative_samples=5  # Default value for negative samples
            )
            
            # Train the model with specified epochs and batch size
            model.train(epochs=self.epochs, batch_size=1024)
            print("\n")
            
            # Create the embeddings array for this scale
            scale_embeddings = np.zeros((num_of_nodes, self.embed_dim))
            for node in original_graph.nodes():
                node_str = str(node)
                vector = model.get_word_vector(node_str)
                if vector is not None:
                    scale_embeddings[self.node_to_idx[node_str]] = vector
            
            # Store this scale's embeddings
            self.scale_embeddings[scale] = scale_embeddings
    
    def get_scale_embedding(self, scale: int) -> np.ndarray:
        """
        The method returns embeddings for a specific scale.
        
        Args:
            scale (int): The scale for which to get embeddings
            
        Returns:
            np.ndarray: A 2D array of node embeddings for the specified scale
            
        Raises:
            ValueError: If the model has not been fitted or if the scale is not available
        """
        if not self.scale_embeddings:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        if scale not in self.scales:
            raise ValueError(f"Scale {scale} not found. Available scales: {self.scales}")
        
        return self.scale_embeddings[scale]
    
    def get_scale_embedding_specific(self, nodes: List[Any], scale: int) -> np.ndarray:
        """
        The method gets embeddings only for the specified list of nodes at a specific scale.
    
        Args:
            nodes (List[Any]): List of node IDs (the IDs as they are in the graph)
            scale (int): The scale to get embeddings for
        
        Returns:
            np.ndarray: Embeddings for the specified nodes at the specified scale.
                        Shape (len(nodes), embed_dim)
                   
        Raises:
            ValueError: If the model has not been fitted or if the scale is not available
        """
        if not self.scale_embeddings or not self.scales:
            raise ValueError("Model has not been fitted. Call fit() first.")
    
        if scale not in self.scales:
            raise ValueError(f"Scale {scale} not found. Available scales: {self.scales}")
    
        # Get the embeddings for the specified scale
        scale_embeddings = self.scale_embeddings[scale]
    
        # Get the indices for the specified nodes
        idxs = [self.node_to_idx[str(node)] for node in nodes]
    
        # Return the embeddings for the specified nodes at the specified scale
        return scale_embeddings[idxs]
    
    def store_emb(self, path: str):
        """
        Stores the embeddings and node mapping to files.
        Args:
            path (str): Path where to save the embeddings
        """
        if not self.scale_embeddings or not self.scales:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save individual scale embeddings
        for scale in self.scales:
            scale_path = os.path.join(os.path.dirname(path), os.path.basename(path) + f"_scale_{scale}_embeddings.npy")
            np.save(scale_path, self.scale_embeddings[scale])
        
        # Save scales list
        scales_path = os.path.join(os.path.dirname(path), os.path.basename(path) + "_scales.json")
        with open(scales_path, 'w') as f:
            json.dump(self.scales, f)
        
        # Save node mapping
        mapping_path = os.path.join(os.path.dirname(path), os.path.basename(path) + "_mapping.json")
        with open(mapping_path, 'w') as f:
            json.dump({str(k): int(v) for k, v in self.node_to_idx.items()}, f)
    
    def load_emb(self, path: str) -> Tuple[Dict[str, int], List[int], Dict[int, np.ndarray]]:
        """
        Loads embeddings, node mapping, and scales from files.
        Args:
            path (str): Path to the saved embeddings (without file extensions)
        Returns:
            Tuple[Dict[str, int], List[int], Dict[int, np.ndarray]]: 
                - Node to index mapping
                - List of scales
                - Dictionary of scale-specific embeddings
        """
        # Construct paths
        mapping_path = os.path.join(os.path.dirname(path), os.path.basename(path) + "_mapping.json")
        scales_path = os.path.join(os.path.dirname(path), os.path.basename(path) + "_scales.json")

        if not os.path.exists(mapping_path) or not os.path.exists(scales_path):
            raise FileNotFoundError("Mapping or scales file not found.")
    
        # Load node mapping
        with open(mapping_path, 'r') as f:
            self.node_to_idx = json.load(f)
            self.node_to_idx = {k: int(v) for k, v in self.node_to_idx.items()}
    
        # Load scales
        with open(scales_path, 'r') as f:
            self.scales = json.load(f)
    
        # Load scale-specific embeddings
        for scale in self.scales:
            scale_path = os.path.join(os.path.dirname(path), os.path.basename(path) + f"_scale_{scale}_embeddings.npy")
            if os.path.exists(scale_path):
                self.scale_embeddings[scale] = np.load(scale_path)
            else:
                raise FileNotFoundError(f"Embeddings for scale {scale} not found")