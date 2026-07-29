# 1. What is a GNN?

A Graph Neural Network (GNN) is a type of machine learning model built
specifically to analyze data structured as a **graph**---meaning a web
of **nodes** (the entities) connected by **edges** (the relationships).

# 2. Why GNNs? (The Problem with Traditional ML)

Algorithms like Random Forests or SVMs are rigid. They require data to
be fed to them in flat, 2D spreadsheets (rows and columns).

However, the real world doesn\'t operate in flat spreadsheets Example:

-   **Social Networks:** People connected by friendships.

-   **Chemistry:** Atoms connected by molecular bonds.

-   **Traffic Systems:** Cities connected by highways.

If you force this interconnected web into a flat spreadsheet, you
permanently destroy the information about *how* everything is connected.
GNNs were invented because we needed an algorithm that could
mathematically read and learn from the **relationships** between data
points, not just the data points themselves.

# 3. How do they work?

They use **\"Message Passing.\"** Every node looks at its direct
neighbors, gathers their data, and combines it with its own data to
understand its place in the broader network

# 4. Core Intuition: From DSA to Matrix Math

To process graphs efficiently, we must translate standard Data
Structures & Algorithms (DSA) concepts like adjacency lists into linear
algebra. A graph is represented by two primary matrices:

-   **Adjacency Matrix (A):** An N × N matrix mapping the edges (e.g.,
    Aᵢⱼ = 1 if node i connects to j).

-   **Feature Matrix (X):** An N × F matrix where each row represents a
    node, and columns represent its features (e.g., voltage, frequency
    bands).

**The Normalization Trick:** Multiplying A by X only sums a node\'s
neighbors, ignoring the node\'s own data. We mathematically force
\"self-loops\" by adding an identity matrix: Ã = A + I. We then
normalize this by the node degrees to prevent highly-connected hub nodes
from dominating the signal, resulting in the standard propagation
matrix: D̃⁻¹/² Ã D̃⁻¹/².

# 5. The Engine: Message Passing Neural Networks (MPNN)

GNNs operate on a \"Message Passing\" framework, which is conceptually
identical to running a parallelized Breadth-First Search (BFS) across
the entire graph. Each layer consists of three steps:

1.  **Message:** Node j projects its features through a learned weight
    matrix to create a message for node i.

2.  **Aggregate:** Node i collects messages from its immediate
    neighbors. This function **must be permutation-invariant** (e.g.,
    Mean, Sum, or Max) because graph neighbors have no canonical
    ordering.

3.  **Update:** Node i combines the aggregated neighbor messages with
    its own previous state to form its new hidden state.

**Geometrical Receptive Field:** 1 layer of message passing = 1-hop BFS
radius. Stacking K layers allows a node to understand its K-hop
neighborhood.

# 6. Why CNNs Fail on Graph Data

Convolutional Neural Networks (CNNs) are completely fundamentally broken
when applied to irregular topologies. CNN filters rely on Euclidean
data, which requires:

-   **Fixed neighborhoods:** Every pixel has exactly 8 neighbors.

-   **Consistent ordering:** Neighbors are always Top, Bottom, Left,
    Right.

-   **Translation equivariance:** A 3x3 filter behaves identically
    anywhere on the grid.

Graph data (like the brain) is non-Euclidean. A node might have 1
neighbor or 50, and there is no concept of a \"left\" neighbor. GNNs
decouple the neighborhood lookup (using the Adjacency Matrix) from the
feature aggregation, allowing them to process arbitrary, dynamic
topologies.

# 7. Key Architectures: GCN vs. GAT

There are two dominant paradigms for deciding how to weight the incoming
messages from neighbors:

-   **Graph Convolutional Networks (GCN):** Weights are structurally
    fixed. A neighbor\'s importance is strictly determined by its
    mathematical degree (connections). It treats all neighbors
    relatively equally.

-   **Graph Attention Networks (GAT):** Weights are learned from
    content. GAT uses an attention mechanism to calculate an importance
    score (αᵢⱼ) between nodes.

    -   *Note for implementation:* GAT is vastly superior for noisy
        biological networks because it learns to highly weight critical
        functional connections and ignore spurious, noisy edges.

# 8. The Over-smoothing Problem

Unlike CNNs, we cannot stack 50 layers in a GNN. Due to the mathematics
of message passing, repeatedly multiplying the normalized adjacency
matrix acts like a random walk.

As layers approach infinity (L → ∞), every node aggregates data from the
entire graph, and all feature vectors collapse into the exact same
uninformative subspace.

-   **Rule of Thumb:** Keep GNNs to 2-4 layers. Mitigate over-smoothing
    using Residual Connections or DropEdge regularizations.

# 9. Real-World Mapping: EEG Brain Networks

To classify cognitive states or anomalies (like seizures) using this
architecture, the physical brain recording is directly mapped to a graph
topology:

-   **The Nodes (V):** The physical EEG electrodes (64 to 256 channels).
    The feature vector (X) contains the time-series voltage or frequency
    power bands.

-   **The Edges (E):** The functional connectivity between regions.
    Edges are weighted by statistical correlations (e.g., Pearson
    correlation or phase-locking values) between different electrodes.

-   **The Pipeline:** We run 2-3 layers of GAT message passing to encode
    the spatial relationships, followed by a Global Pooling (Readout)
    layer that collapses the entire graph into a single vector. A
    standard MLP classification head then outputs the probability of the
    brain state (e.g., Seizure vs. Normal).

**IMPLEMENTATION :**

Link:

[[GNN.ipynb]{.underline}](https://colab.research.google.com/drive/1sUoH-QmwvpNB6ZpPEJFg9EMJtlVMf9ot?usp=sharing)

import torch

import torch.nn.functional as F

from torch_geometric.datasets import EllipticBitcoinDataset

from torch_geometric.nn import GATConv

\# 2. Download and Load the Dataset

print(\"Downloading Elliptic Bitcoin Dataset\...\")

dataset = EllipticBitcoinDataset(root=\'./data/Elliptic\')

data = dataset\[0\]

\# 3. Create Strict Masks (Fixing the Data Leakage)

\# The dataset provides a chronological split, but we must explicitly

\# filter out unlabelled nodes (class 2) from both masks.

train_mask = data.train_mask & (data.y != 2)

test_mask = data.test_mask & (data.y != 2)

\# 4. Define the GNN Architecture

class AMLModel(torch.nn.Module):

def \_\_init\_\_(self, in_channels, hidden_channels, out_channels):

super().\_\_init\_\_()

self.conv1 = GATConv(in_channels, hidden_channels)

self.conv2 = GATConv(hidden_channels, out_channels)

def forward(self, x, edge_index):

x = self.conv1(x, edge_index)

x = F.relu(x)

x = F.dropout(x, p=0.5, training=self.training)

x = self.conv2(x, edge_index)

return x

device = torch.device(\'cuda\' if torch.cuda.is_available() else
\'cpu\')

model = AMLModel(dataset.num_features, 64, 2).to(device)

data = data.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01,
weight_decay=5e-4)

\# 5. Train the Model

print(\"\\nStarting Training Loop\...\")

model.train()

for epoch in range(11):

optimizer.zero_grad()

out = model(data.x, data.edge_index)

\# LOSS IS ONLY CALCULATED ON THE TRAINING SET (Time steps 1-34)

loss = F.cross_entropy(out\[train_mask\], data.y\[train_mask\])

loss.backward()

optimizer.step()

if epoch % 10 == 0:

print(f\'Epoch {epoch:\>3} \| Loss: {loss:.4f}\')

\# 6. Evaluate and Print

model.eval()

with torch.no_grad():

out = model(data.x, data.edge_index)

\_, pred = out.max(dim=1)

\# Accuracy is calculated ONLY on the UNSEEN test set (Time steps 35-49)

correct = int(pred\[test_mask\].eq(data.y\[test_mask\]).sum().item())

accuracy = correct / int(test_mask.sum())

print(f\'\\nFinal Accuracy on UNSEEN Test Data: {accuracy:.2%}\')

print(\"\\n=== SAMPLE PREDICTIONS (Unseen Data) ===\")

\# Grab indices from the test_mask to prove no data leakage

test_indices = test_mask.nonzero(as_tuple=True)\[0\].cpu().numpy()

\# Loop through a controlled 20-row sample

for idx in test_indices\[:20\]:

predicted_class = pred\[idx\].item()

actual_class = data.y\[idx\].item()

\# CORRECTED MAPPING: 0 = Licit, 1 = Illicit

pred_text = \"ILLICIT (Fraud)\" if predicted_class == 1 else \"LICIT
(Safe)\"

actual_text = \"ILLICIT (Fraud)\" if actual_class == 1 else \"LICIT
(Safe)\"

status = \"✅ MATCH\" if predicted_class == actual_class else \"❌
MISMATCH\"

print(f\"Transaction ID: {idx:\<6} \| Predicted: {pred_text:\<15} \|
Actual: {actual_text:\<15} \| {status}\")



**Explanation of this Implementation:**

This GAT implementation uses the Elliptic Bitcoin dataset, which
contains a directed graph of over 200,000 cryptocurrency transactions,
with 46,564 labeled transactions belonging to 2 classes: Licit (Safe)
and Illicit (Money Laundering). The input to the model is the entire
transaction network, where each node has 166 numerical features and
edges represent the flow of funds. During training, the graph first
passes through a Graph Attention (\`GATConv\`) layer that aggregates
structural features from neighboring transactions while mathematically
weighing suspicious connections, followed by a ReLU activation layer
that removes negative values and Dropout to prevent overfitting. The
updated node embeddings are passed through a second \`GATConv\` layer
that generates 2 output scores per node. The model is trained for 50
epochs on a chronological training split (time steps 1--34). After
training, the GNN predicts the class of unseen test transactions (time
steps 35--49) by selecting the class with the highest raw score.

**Input:**

Elliptic Bitcoin transaction graph

Node features: 166 (local and structural data)

Edges: Directed flow of funds between nodes

Input tensor shape: \`x: \[N, 166\]\` (Features), \`edge_index: \[2,
E\]\` (Connections)

**Output:**

A vector of 2 scores (logits) corresponding to the 2 classes:

\[Licit (0), Illicit (1)\]

The \`.max(dim=1)\` function evaluates these raw scores.

The class with the highest score is selected as the final prediction for
that specific node.

**Example Output:**

Licit (Safe) = 11%

Illicit (Fraud) = 89%

Final Prediction = Illicit
