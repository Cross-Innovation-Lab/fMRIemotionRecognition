import torch
import torch.nn as nn
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims
    def forward(self, x):
        return x.permute(*self.dims)

class TransformerClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        d_model=64,          # token 嵌入维度
        nhead=4,             # 多头注意力头数
        num_layers=2,        # Transformer 编码器层数
        dim_feedforward=256, # 前馈网络维度
        dropout=0.1,         # dropout 比例
        num_classes=6,       # 输出类别数
        lr=1e-3,             # 学习率
        max_epochs=100,      # 最大训练轮数
        batch_size=32,       # 批大小
        device='cuda' if torch.cuda.is_available() else 'cpu',
        random_state=42
    ):
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.num_classes = num_classes
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.device = device
        self.random_state = random_state

    def _build_model(self, seq_len):
        self.model = nn.Sequential(
            # 将每个标量体素映射到 d_model 维
            nn.Linear(1, self.d_model),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.d_model,
                    nhead=self.nhead,
                    dim_feedforward=self.dim_feedforward,
                    dropout=self.dropout,
                    batch_first=True,
                    activation='gelu'
                ),
                num_layers=self.num_layers
            ),
            Permute(0,2,1),
            # 平均池化所有 token，然后分类
            nn.AdaptiveAvgPool1d(1),   # 输入 shape (batch, d_model, seq_len) -> (batch, d_model, 1)
            nn.Flatten(),
            nn.Linear(self.d_model, self.num_classes)
        )

    def fit(self, X, y):
        X, y = check_X_y(X, y, ensure_2d=True)
        self.n_features_in_ = X.shape[1]  # 体素数量 = seq_len
        self.classes_ = np.unique(y)
        if self.num_classes is None:
            self.num_classes = len(self.classes_)

        rng = np.random.RandomState(self.random_state)
        torch.manual_seed(self.random_state)

        self._build_model(self.n_features_in_)
        self.model.to(self.device)

        X_tensor = torch.from_numpy(X).float().unsqueeze(-1)  # (n, seq_len, 1)
        y_tensor = torch.from_numpy(y).long()

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(self.max_epochs):
            total_loss = 0.0
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(batch_X)  # (b, num_classes)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            # 可选：加入早停，此处省略以保持简洁
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X, ensure_2d=True)
        X_tensor = torch.from_numpy(X).float().unsqueeze(-1).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            preds = logits.argmax(dim=1).cpu().numpy()
        return preds

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X, ensure_2d=True)
        X_tensor = torch.from_numpy(X).float().unsqueeze(-1).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs