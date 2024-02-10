import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        # Using ResNet-34 for a lighter model
        resnet = models.resnet18(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        # TODO: Complete this function
        self.lstm = nn.LSTM(embed_size,  hidden_size, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        self.embed = nn.Embedding(vocab_size, embed_size)
        
    def forward(self, features, captions):
        embeddings = self.embed(captions[:, :-1])  # Exclude the <end> token
        #embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        # TODO: Complete this function
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        features = features.unsqueeze(1)
        vals = torch.cat((features, embeddings), dim=1)
        h, _ = self.lstm(vals)
        #out, self.hidden_state = self.lstm(vals, self.hidden_state)
        out = self.fc_out(h)
        return out

    def sample(self, inputs, states=None, max_len=20):
        "accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len)"
        predicted_sentence = []
        for i in range(max_len):
            h, states = self.lstm(inputs, states)
            out = self.fc_out(h.squeeze(1))
            predicted = out.max(1)[1]
            print(predicted)
            predicted_sentence.append(predicted.tolist()[0])
            inputs = self.embed(predicted).unsqueeze(1)
            if (predicted == 1):
                # We predicted the <end> word, so there is no further prediction to do
                break
        return predicted_sentence
    
