from typing import List, Any
import networkx as nx
import numpy as np
import os
import json
import sys
# Remove gensim import and use custom Word2Vec
parent_dir = os.path.abspath(os.path.join(".."))
if not parent_dir in sys.path:
    sys.path.append(parent_dir)
from utils.word2vec import Word2Vec  # Import your custom Word2Vec implementation

class RandomWalkEmbedder:
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
        self.embeddings = None
        self.node_to_idx = None
        np.random.seed(self.seed)  # Set the random seed

    @staticmethod
    def walk_node(graph: nx.Graph, node: Any, walk_length: int) -> List[Any]:
        """
        The method generates a single random walk starting from a given node in the graph.
        Returns:
            walk (List[Any]): A list of nodes representing the path taken during the random walk.
        Args:
            graph (nx.Graph): The graph on which the random walk will be performed.
            node (Any): The starting node for the random walk.
            walk_length (int): The total number of nodes (steps) the random walk should have, including the starting node.
        """
        walk = [node]
        for _ in range(walk_length - 1):
            neighbors = list(graph.neighbors(walk[-1]))
            if neighbors:
                walk.append(np.random.choice(neighbors))
        return walk

    @staticmethod
    def walk_graph(graph: nx.Graph, walks_per_node: int, walk_length: int) -> List[List[Any]]:
        """
        The method generates multiple random walks for each node in the graph.
        Returns:
            walks (List[List[Any]]): A list of walks, where each walk is a list of nodes visited in order.
        Args:
            graph (nx.Graph): The graph on which the random walks will be performed.
            walks_per_node (int): Number of random walks to generate starting from each node.
            walk_length (int): The total number of nodes (steps) in each random walk, including the starting node.
        """
        walks = []
        nodes = list(graph.nodes())
        for node in nodes:
            for _ in range(walks_per_node):
                walk = RandomWalkEmbedder.walk_node(graph, node, walk_length)
                walks.append([str(n) for n in walk])  # Convert nodes to strings for Word2Vec
        return walks

    def fit(self, original_graph: nx.classes.graph.Graph = nx.karate_club_graph()):
        """
        The method fits the Random Walk model to the graph.
        It will create first, the random walks on the graph provided in the constructor.
        Then it will train the Word2Vec model with the dataset created, storing after the train phase
        the embeddings in the self.embeddings atribute for each node of the graph given in the constructor.
        Returns:
            nothing
        Args:
            original_graph (nx.Graph): The original graph to be used for embedding. Default is the karate club graph.
        """
        # Initialize node to index mapping
        self.node_to_idx = {str(node): idx for idx, node in enumerate(original_graph.nodes())}
        num_of_nodes = original_graph.number_of_nodes()

        # Generate random walks using the static method from RandomWalkEmbedder
        walks = RandomWalkEmbedder.walk_graph(original_graph, self.walk_number, self.walk_length)
        print(f"Generated a total of {len(walks)} random walks of length {self.walk_length},starting from {len(walks)/self.walk_number} nodes.\n")

        # Train custom Word2Vec model
        # Adapt parameters to match the custom Word2Vec implementation
        model = Word2Vec(
            sequences=walks,
            embedding_dim=self.embed_dim,
            window_size=self.window_size,
            negative_samples=5  # Default value for negative samples
        )
        
        # Train the model with specified epochs and batch size
        model.train(epochs=self.epochs, batch_size=1024)
        print("\n")
        
        # Create the embeddings array using get_word_vector method
        self.embeddings = np.zeros((num_of_nodes, self.embed_dim))
        for node in original_graph.nodes():
            node_str = str(node)
            vector = model.get_word_vector(node_str)
            if vector is not None:
                self.embeddings[self.node_to_idx[node_str]] = vector

    def get_embedding(self) -> np.ndarray:
        """
        The method returns the full node embeddings after the model has been fitted.
        Returns:
            np.ndarray: A 2D array of node embeddings with shape (number of nodes, embedding dimension).
        Raises:
            ValueError: If the model has not been fitted yet and embeddings are not available.
        """
        if self.embeddings is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        return self.embeddings
    
    def get_embedding_specific(self, nodes: List[Any]) -> np.ndarray:
        """
        The method gets embeddings only for the specified list of nodes.
        Args:
            nodes: list of node IDs (the IDs as they are in the graph)
        Returns:
            np.ndarray: Embeddings for the specified nodes. Shape (len(nodes), embed_dim)
        """
        if self.embeddings is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        idxs = [self.node_to_idx[str(node)] for node in nodes]
        return self.embeddings[idxs]
    
    def store_emb(self, path: str):
        """
        Stores the embeddings and node mapping to a file.
        Args:
            path (str): Path where to save the embeddings
        """
        if self.embeddings is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save embeddings and mapping to the specified path
        embeddings_path = os.path.join(os.path.dirname(path), os.path.basename(path) + "_embeddings.npy")
        mapping_path = os.path.join(os.path.dirname(path), os.path.basename(path) + "_mapping.json")
        
        # Save embeddings as numpy array
        np.save(embeddings_path, self.embeddings)
        
        # Save node mapping as JSON
        with open(mapping_path, 'w') as f:
            json.dump({str(k): int(v) for k, v in self.node_to_idx.items()}, f)
    
    def load_emb(self,path: str):
        """
        Loads embeddings and node mapping from files.
        Args:
            path (str): Path to the saved embeddings (without file extensions)
        Returns:
            RandomWalkEmbedder: A new instance with loaded embeddings and mapping
        """
        # Construct the full paths
        embeddings_path = os.path.join(os.path.dirname(path), os.path.basename(path) + "_embeddings.npy")
        mapping_path = os.path.join(os.path.dirname(path), os.path.basename(path) + "_mapping.json")

        if not os.path.exists(embeddings_path) or not os.path.exists(mapping_path):
            raise FileNotFoundError("Embeddings or mapping file not found.")
        
        self.embeddings = np.load(embeddings_path)
        
        # Load node mapping
        with open(mapping_path, 'r') as f:
            node_to_idxk = json.load(f)
            self.node_to_idx = {k: int(v) for k, v in node_to_idxk.items()}