import torch
import torch.nn as nn
# from torchsummary import summary


device = "cuda" if torch.cuda.is_available() else "cpu"


class EmotionClassifier(nn.Module):
    def __init__(self):
        super(EmotionClassifier, self).__init__()

        self.core_layer_1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 32, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.MaxPool2d(2),
            nn.Dropout(0.2)
        )

        self.core_layer_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 32, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.MaxPool2d(2),
            nn.Dropout(0.2),
        )

        self.dense_layer = nn.Sequential(
            nn.Flatten(),

            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.2),

            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),

            nn.Linear(512, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),

            nn.Linear(32, 7),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.core_layer_1(x)
        x = self.core_layer_2(x)
        x = self.dense_layer(x)
        return x


if __name__ == "__main__":
    model = EmotionClassifier()

    x = torch.zeros(1, 48, 48)
    print(x.shape)

    # summary(model.to(device), x.to(device).shape)
