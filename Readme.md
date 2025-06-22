🛡️ TrustGraph: Heterogeneous GNN for Marketplace Fraud Detection

TrustGraph is an AI-powered fraud detection framework that leverages Heterogeneous Graph Neural Networks (HeteroGNNs) to detect fake reviews, suspicious sellers, and coordinated abuse across IP addresses in an e-commerce platform. It uses relational reasoning across multiple node types to output TrustScores for reviews, IPs, and sellers.

—

📌 Features

✅ Multimodal Node Features:

Review Nodes: BERT embeddings, TF-IDF vectors, sentiment, rating, verified flag
Product Nodes: Description, price bin, return rate, category
Seller Nodes: Rating, flags, account age
IP Nodes: VPN flag, usage pattern, review count
✅ Graph Construction:

Heterogeneous graph with nodes: review, product, seller, IP
Edge types:
(review) —written_for→ (product)
(product) —sold_by→ (seller)
(review) —sent_from→ (IP)
(review) —similar_to→ (review)
✅ Model Architecture:

Built with PyTorch Geometric (PyG)
HeteroConv with relation-specific aggregation
2-layer GNN for multi-hop message passing
Outputs TrustScore (fraud probability) for each review, IP, and seller
✅ Real-Time Interface:

Interactive frontend (Streamlit or Web App)
Inputs: review text, rating, IP, seller, product
Outputs: TrustScore with fraud likelihood and graph-based explanation
—

🗂️ Project Structure

.
├── data/ # Input CSVs: reviews, products, sellers, IPs
├── preprocess/ # Feature generation scripts (BERT, TF-IDF, etc.)
├── model/ # HeteroGNN model and training code
├── app/ # Inference + web interface (FastAPI / Streamlit)
├── outputs/ # Saved models, trust scores
└── README.md # This file

—

🚀 How It Works

Data Ingestion:
Loads review, product, seller, and IP datasets.
Extracts node-level features.
Graph Building:
Builds a heterogeneous graph in PyTorch Geometric’s HeteroData format.
Computes similarity edges based on review text (BERT + TF-IDF cosine similarity).
Model Training:
Trains a 2-layer HeteroGNN on labeled reviews, sellers, and IPs.
Minimizes binary cross-entropy loss.
Inference:
Given new review + metadata, the model predicts TrustScores for review, IP, and seller.
Optionally visualizes the neighborhood and influencing nodes.
—

📦 Inputs

Each node type has its own CSV:

reviews.csv: review_id, text, rating, verified, timestamp, product_id, ip_id
products.csv: product_id, description, price, return_rate, category
sellers.csv: seller_id, avg_rating, flags, account_age
ips.csv: ip_id, frequency, vpn_flag, hour_profile
—

🧠 Sample Output

Input:

Review: "Great product!"
Rating: 5
Product: P1
Seller: S1
IP: IP1
Output:

TrustScore (Review): 0.91 ⚠️
TrustScore (IP): 0.95 🔴
TrustScore (Seller): 0.82 🟠
Explanation: IP used by 3 flagged reviews, seller linked to high return rate
—

📊 Evaluation Metrics

Precision, Recall, F1 Score
ROC-AUC
Explanation quality via neighborhood tracing
—

💡 Applications

E-commerce trust & safety
Counterfeit detection
Review authenticity scoring
Seller ranking and fraud ring detection
—

📁 Future Work

Integrate GAT-based attention for explainability
Visual TrustGraph explorer UI
Active learning loop for IP/Seller labeling
Real-time deployment on AWS (EKS + SageMaker)
