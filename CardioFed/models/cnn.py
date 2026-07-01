import torch.nn as nn

class CNNFeatureExtractor(nn.Module):

    def __init__(self):

        super(CNNFeatureExtractor, self).__init__()

        self.features = nn.Sequential(

            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1))

        )

        self.fc = nn.Linear(32, 13)

    def forward(self, x):

        x = self.features(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x