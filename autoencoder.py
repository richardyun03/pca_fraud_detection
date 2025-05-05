import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import numpy as np

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def autoencoder_classifier(x_train, y_train, x_test, y_test, encoding_dim=32, epochs=20, batch_size=32, device='cpu'):
    # Ensure numpy arrays
    x_train = np.array(x_train, dtype=np.float32)
    x_test = np.array(x_test, dtype=np.float32)

    # Normalize
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    input_dim = x_train.shape[1]
    model = Autoencoder(input_dim, encoding_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Convert data to torch tensors
    x_train_tensor = torch.tensor(x_train_scaled).to(device)
    x_test_tensor = torch.tensor(x_test_scaled).to(device)

    # Train autoencoder
    model.train()
    for epoch in range(epochs):
        permutation = torch.randperm(x_train_tensor.size(0))
        for i in range(0, x_train_tensor.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_x = x_train_tensor[indices]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_x)
            loss.backward()
            optimizer.step()

    # Extract encoded features
    model.eval()
    with torch.no_grad():
        x_train_encoded = model.encoder(x_train_tensor).cpu().numpy()
        x_test_encoded = model.encoder(x_test_tensor).cpu().numpy()

    # Train classifier on encoded features
    clf = LogisticRegression()
    clf.fit(x_train_encoded, y_train)
    y_pred_proba = clf.predict_proba(x_test_encoded)[:, 1]

    # ROC & AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    

    return fpr, tpr, y_pred_proba, roc_auc, clf
