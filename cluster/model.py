from transformers import ASTFeatureExtractor, ASTModel
from torch import nn
import torch

processor = ASTFeatureExtractor.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTModel.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593")

# build 2 types of models (pure AST and AST + handcrafted features- MFCC)


class ASTHandcrafted(nn.Module):
    def __init__(self, num_classes):
        super(ASTHandcrafted, self).__init__()
        self.ast = model
        self.fc1 = nn.Linear(1280 + 20, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x, mfcc):
        x = self.ast(x)
        x = torch.cat((x, mfcc), 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def freeze(self):
        for param in self.ast.parameters():
            param.requires_grad = False
        for param in self.fc1.parameters():
            param.requires_grad = False
        for param in self.fc2.parameters():
            param.requires_grad = False

# AST + MFCC features (combine with attention)


class ASTHandcraftedAttention(nn.Module):
    def __init__(self, num_classes):
        super(ASTHandcraftedAttention, self).__init__()
        self.ast = model
        self.fc1 = nn.Linear(1280 + 20, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def attention_weighted_sum(self, f1, f2):
        f1 = torch.softmax(f1, 1)
        f2 = torch.softmax(f2, 1)
        f1 = torch.mul(f1, f2)
        return f1

    def forward(self, x, mfcc):
        x = self.ast(x)
        x = self.attention_weighted_sum(x, mfcc)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
