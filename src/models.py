"""Model architectures used by the notebook experiments.

Copied verbatim (whitespace normalised) from the v0 notebooks in ``v0_experiments/``.

The notebooks define per-dataset aliases (``MLP1D_12k``, ``AE1D_12k``, ...) that are
byte-identical to the base classes apart from indentation, so only the base classes
live here; :data:`MODEL_ALIASES` records the alias each notebook used.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x1(in_planes, out_planes, stride=1):
    """3x1 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class MLP1D(nn.Module):
    def __init__(self, input_length: int = 12000, num_classes: int = 4):
        super().__init__()

        # Camada adicional para adaptar a entrada de 12000 para 1024 progressivamente
        self.adaptation_layers = nn.Sequential(
            # 12000 -> 4096
            nn.Linear(input_length, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.7), # Dropout alto para evitar overfitting na entrada

            # 4096 -> 2048
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            # 2048 -> 1024 (Conecta com a arquitetura original)
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4)
        )

        # Arquitetura original:
        self.fc3 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )

        self.fc4 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )

        self.fc5 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )

        self.fc6 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )

        self.fc7 = nn.Sequential(
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        out = torch.flatten(x, 1)

        # Passa pela adaptação primeiro
        out = self.adaptation_layers(out)

        # Segue o fluxo normal
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.fc5(out)
        out = self.fc6(out)
        out = self.fc7(out)

        return out


class AE1D(nn.Module):
    """
    Implementação do Autoencoder 1D Adaptado para 12k pontos.
    Arquitetura: 12000 -> 512 -> 256 -> 128 -> 64 (Latent) -> 128 -> 256 -> 512 -> 12000.
    """
    def __init__(self, input_length: int = 12000, latent_dim: int = 64, num_classes: int = 4, dropout_rate: float = 0.4):
        super(AE1D, self).__init__()

        # --- Encoder ---
        self.encoder = nn.Sequential(
            # Camada 1: Compressão Direta (12000 -> 512)
            nn.Linear(input_length, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            # Camada 2: 512 -> 256
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            # Camada 3: 256 -> 128
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            # Camada Latente: 128 -> 64
            nn.Linear(128, latent_dim)
        )

        # --- Decoder ---
        # Simétrico ao encoder
        self.decoder = nn.Sequential(
            # Latente -> 128
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            # 128 -> 256
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            # 256 -> 512
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            # Reconstrução Final: 512 -> 12000
            nn.Linear(512, input_length)
        )

        # Classificador (Fine-tuning)
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        # Flatten para garantir (Batch, 12000)
        x = torch.flatten(x, 1)

        # Encoder
        latent_features = self.encoder(x)

        # Decoder (Reconstrução do sinal de 12k)
        reconstruction = self.decoder(latent_features)

        # Classifier
        classification_output = self.classifier(latent_features)

        return classification_output, reconstruction, latent_features


class SAE1D(nn.Module):
    """
    Implementação do Sparse Autoencoder 1D adaptado para 12k pontos.
    Arquitetura: 12000 -> 512 -> 256 -> 128 -> 64 (Latent).
    """
    def __init__(self, input_length: int = 12000, latent_dim: int = 64, num_classes: int = 4):
        super(SAE1D, self).__init__()

        # --- Encoder ---
        self.encoder = nn.Sequential(
            # Camada 1: Compressão Direta (12000 -> 512)
            # Redução drástica necessária para viabilidade
            nn.Linear(input_length, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2), # Adicionado Dropout leve

            # Camada 2: 512 -> 256
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            # Camada 3: 256 -> 128
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            # Camada Latente: 128 -> Latent Dim
            nn.Linear(128, latent_dim)
        )

        # A Sigmoid é OBRIGATÓRIA para SAE se você usar KL Divergence Loss
        # Ela força os neurônios latentes a ficarem entre [0, 1] (probabilidade de ativação)
        self.sparsity_activation = nn.Sigmoid()

        # --- Decoder ---
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            # Reconstrução: 512 -> 12000
            nn.Linear(512, input_length)
        )

        # Classificador (Fine-tuning)
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        # Garante (Batch, 12000)
        x = torch.flatten(x, 1)

        # 1. Codificação Linear
        features = self.encoder(x)

        # 2. Ativação Esparsa (Latent Space)
        # Importante: A saída aqui estará entre 0 e 1
        latent_features = self.sparsity_activation(features)

        # 3. Reconstrução
        reconstruction = self.decoder(latent_features)

        # 4. Classificação
        classification_output = self.classifier(latent_features)

        return classification_output, reconstruction, latent_features


class DAE1D(nn.Module):
    """
    Implementação do Denoising Autoencoder (DAE) adaptado para 12k pontos.
    Arquitetura: 12000 -> 512 -> 256 -> 128 -> 64 (Latent).
    """
    def __init__(self, input_length: int = 12000, latent_dim: int = 64, num_classes: int = 4, noise_factor: float = 0.5):
        super(DAE1D, self).__init__()
        self.noise_factor = noise_factor

        # Encoder
        self.encoder = nn.Sequential(
            # Camada 1: Compressão Direta (12000 -> 512)
            nn.Linear(input_length, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            # Camada 2: 512 -> 256
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            # Camada 3: 256 -> 128
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            # Latent
            nn.Linear(128, latent_dim)
        )

        # Decoder
        # Simétrico
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            # Reconstrução: 512 -> 12000
            nn.Linear(512, input_length)
        )

        # --- Classificador ---
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        # (Batch, 12000)
        x = torch.flatten(x, 1)

        # Denoising Injection
        if self.training:
            # Adiciona ruído apenas durante o treino
            noise = torch.randn_like(x) * self.noise_factor
            x_noisy = x + noise
        else:
            x_noisy = x

        # Encoder
        latent_features = self.encoder(x_noisy)

        # Decoder
        reconstruction = self.decoder(latent_features)

        # Classifier
        classification_output = self.classifier(latent_features)

        # (Classification, Reconstruction, Latent[Opcional])
        return classification_output, reconstruction, latent_features


class CNN1D(nn.Module):
    def __init__(self, input_length: int, num_classes: int):
        """
        1D CNN for vibration signal classification.
        Args:
            input_length: length of the input signal
            num_classes: number of output classes
        """
        super(CNN1D, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.AdaptiveMaxPool1d(16)  # reduce dynamically to fixed size

        # compute flattened size
        example_input = torch.zeros(1, 1, input_length)  # [B, C, L]
        with torch.no_grad():
            x = self.pool1(F.relu(self.bn1(self.conv1(example_input))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            x = self.pool3(F.relu(self.bn3(self.conv3(x))))
            flattened_size = x.shape[1] * x.shape[2]

        self.fc1 = nn.Linear(flattened_size, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: [B, L] or [B, 1, L]
        if x.ndim == 2:
            x = x.unsqueeze(1)  # add channel dim

        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class LeNet1D(nn.Module):
    def __init__(self, in_channel=1, out_channel=4):
        super(LeNet1D, self).__init__()

        # --- Bloco Convolucional 1 ---
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channel, 6, kernel_size=64, stride=4, padding=30),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), # 3000 -> 1500
        )

        # --- Bloco Convolucional 2 ---
        self.conv2 = nn.Sequential(
            nn.Conv1d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(12)
        )

        # --- Classificador (Fully Connected) ---
        # Input achatado: 16 canais * 12 pontos = 192 features
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 12, 120), # Tamanho clássico da LeNet-5
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(120, 84), # Tamanho clássico da LeNet-5
            nn.ReLU()
        )

        self.fc3 = nn.Linear(84, out_channel)

    def forward(self, x):
        # Garante dimensão de canal (Batch, 1, 12000)
        if x.ndim == 2:
            x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.conv2(x)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x1(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x1(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet18(nn.Module):
    """
    Implementação da ResNet-18 adaptada para sinais 1D de 12.000 pontos.
    """
    def __init__(self, input_length=12000, in_channel=1, num_classes=10):
        super(ResNet18, self).__init__()

        # Configuração padrão da ResNet18
        block = BasicBlock
        layers = [2, 2, 2, 2] # 2 blocos por camada = 18 layers total

        self.inplanes = 64

        # STEM (Camada de Entrada)
        # Input: (Batch, 1, 12000)
        self.conv1 = nn.Conv1d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Blocos Residuais
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Classificador
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Inicialização de Pesos
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Tratamento de Dimensão: Garante (Batch, 1, Length)
        if x.ndim == 2:
            x = x.unsqueeze(1)

        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Head
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class AlexNet1D(nn.Module):
    def __init__(self, input_length=12000, in_channel=1, num_classes=10):
        super(AlexNet1D, self).__init__()

        self.features = nn.Sequential(
            # Conv1: Kernel 11 e Stride 4.
            # Reduz entrada 12000 -> ~3000
            nn.Conv1d(in_channel, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2), # 3000 -> 1500

            # Conv2
            nn.Conv1d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2), # 1500 -> 750

            # Conv3
            nn.Conv1d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Conv4
            nn.Conv1d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Conv5
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2), # 750 -> 375
        )

        self.avgpool = nn.AdaptiveAvgPool1d(6)

        self.classifier = nn.Sequential(
            nn.Dropout(),
            # 256 canais * 6 dimensão temporal = 1536 features
            nn.Linear(256 * 6, 1024), # Mantido 1024 conforme seu código (o paper original usa 4096)
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        # Tratamento de segurança para dimensão (Batch, 1, 12000)
        if x.ndim == 2:
            x = x.unsqueeze(1)

        x = self.features(x)
        x = self.avgpool(x)

        # Flatten robusto
        x = torch.flatten(x, 1)

        x = self.classifier(x)
        return x


class BiLSTM(nn.Module):
    def __init__(self, in_channel=1, out_channel=10):
        super(BiLSTM, self).__init__()

        # Hiperparâmetros Internos
        self.hidden_dim = 64
        self.kernel_num = 16
        self.num_layers = 2

        # self.V define o comprimento da sequência que entra na LSTM
        self.V = 100

        # Camada de stem
        self.embed1 = nn.Sequential(
            # Kernel grande e Stride 4 para lidar com alta taxa de amostragem
            nn.Conv1d(in_channel, self.kernel_num, kernel_size=64, stride=4, padding=30),
            nn.BatchNorm1d(self.kernel_num),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4)
        )

        self.embed2 = nn.Sequential(
            nn.Conv1d(self.kernel_num, self.kernel_num*2, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.kernel_num*2),
            nn.ReLU(inplace=True),
            # AdaptiveMaxPool força a saída a ter exatamente comprimento self.V (100) garantindo entrada constante para a LSTM
            nn.AdaptiveMaxPool1d(self.V)
        )

        # Camada Recorrente (BiLSTM)
        # Input Size: kernel_num*2 (32 features por passo de tempo)
        self.bilstm = nn.LSTM(
            input_size=self.kernel_num*2,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            bidirectional=True,
            batch_first=True,
            bias=False
        )

        # Classificador
        # O input da linear é: Comprimento da Sequência (V) * (Hidden * 2 direções)
        self.hidden2label1 = nn.Sequential(
            nn.Linear(self.V * 2 * self.hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.hidden2label2 = nn.Linear(512, out_channel)

    def forward(self, x):
        # (Batch, 1, 12000)
        if x.ndim == 2:
            x = x.unsqueeze(1)

        # Extração de Features Convolucionais
        x = self.embed1(x)
        x = self.embed2(x)
        # (Batch, 32, 100) -> (Batch, Channels, Time)

        # Preparação para LSTM
        # LSTM espera (Batch, Time, Features/Channels)
        x = x.permute(0, 2, 1) # (Batch, 100, 32)

        # Processamento Recorrente
        bilstm_out, _ = self.bilstm(x)
        bilstm_out = torch.tanh(bilstm_out)

        # Classificação
        bilstm_out = bilstm_out.reshape(bilstm_out.size(0), -1)

        logit = self.hidden2label1(bilstm_out)
        logit = self.hidden2label2(logit)

        return logit


#: Model builders, keyed by the short experiment key used across the runner.
MODEL_BUILDERS = {
    "mlp1d": lambda input_length, num_classes: MLP1D(
        input_length=input_length, num_classes=num_classes),
    "ae1d": lambda input_length, num_classes: AE1D(
        input_length=input_length, latent_dim=64, num_classes=num_classes),
    "sae1d": lambda input_length, num_classes: SAE1D(
        input_length=input_length, latent_dim=64, num_classes=num_classes),
    "dae1d": lambda input_length, num_classes: DAE1D(
        input_length=input_length, latent_dim=64, num_classes=num_classes, noise_factor=0.5),
    "cnn1d": lambda input_length, num_classes: CNN1D(
        input_length=input_length, num_classes=num_classes),
    "lenet1d": lambda input_length, num_classes: LeNet1D(
        in_channel=1, out_channel=num_classes),
    "resnet18": lambda input_length, num_classes: ResNet18(
        input_length=input_length, num_classes=num_classes),
    "alexnet": lambda input_length, num_classes: AlexNet1D(
        input_length=input_length, in_channel=1, num_classes=num_classes),
    "bilstm": lambda input_length, num_classes: BiLSTM(
        in_channel=1, out_channel=num_classes),
}

MODEL_KEYS = tuple(MODEL_BUILDERS)


def build_model(key: str, input_length: int, num_classes: int) -> nn.Module:
    """Instantiate the model for ``key`` exactly as the notebooks do."""
    try:
        builder = MODEL_BUILDERS[key]
    except KeyError:
        raise KeyError(f"unknown model key {key!r}; expected one of {MODEL_KEYS}") from None
    return builder(input_length, num_classes)
