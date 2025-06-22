# Amazon Hackathon Season 5 – Trust Score Prediction using Heterogeneous Graph Neural Networks

This repository implements a Heterogeneous Graph Neural Network (MultiTrustGNN) that predicts trustworthiness scores for reviews, IP addresses, and sellers on an e-commerce platform. The model is trained using relational graph data and enriched features such as BERT embeddings, TF-IDF vectors, sentiment scores, and metadata.

---

## 📁 Repository Structure

Amazon-Hackathon-Season-5/
├── Model.py # GNN model architecture definition
├── testing_file.py # Inference script: adds new reviews and updates trust scores
├── Training Dataset/ # CSV files: review_wl.csv, ip_wl.csv, seller_wl.csv, product_wl.csv
├── multi_trust_gnn.pth # Pretrained model weights
├── trust_scores_all_reviews.csv # Output: trust scores of all review nodes
├── trust_scores_all_ips.csv # Output: trust scores of all IP nodes
├── trust_scores_all_sellers.csv # Output: trust scores of all seller nodes
└── README.md # Instructions and documentation

yaml
Copy
Edit

---

## ✅ Installation Requirements

Run the following to install all required packages:

```bash
pip install torch torchvision torchaudio
pip install torch-geometric
pip install transformers
pip install scikit-learn
pip install nltk
Also, download the VADER sentiment lexicon:

python
Copy
Edit
import nltk
nltk.download('vader_lexicon')
🚀 Running the Model in Google Colab or Locally
Step 1: Clone the Repository
bash
Copy
Edit
git clone https://github.com/your-username/Amazon-Hackathon-Season-5.git
cd Amazon-Hackathon-Season-5
Replace your-username with your GitHub username.

Step 2: Prepare Model Weights
Ensure multi_trust_gnn.pth is present in the root directory. If not, upload it manually in Colab:

python
Copy
Edit
from google.colab import files
files.upload()  # Select multi_trust_gnn.pth
Step 3: Run the Inference Pipeline
Execute the testing script to:

Load the pretrained GNN model

Add multiple new review nodes

Fine-tune the model with known labels for the new reviews

Observe how the trust scores of connected IPs and Sellers change

Save updated trust scores to CSV files

bash
Copy
Edit
python testing_file.py
📤 Outputs
After running testing_file.py, the following CSV files will be generated or updated:

trust_scores_all_reviews.csv – Trust scores for all reviews (existing + added)

trust_scores_all_ips.csv – Trust scores for all IP addresses

trust_scores_all_sellers.csv – Trust scores for all sellers

These files include updated scores after adding and fine-tuning on new review nodes.
