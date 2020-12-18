import torch
from Resnet_Model.Resnet import ResNet
from Utils import MnistLoadData
from Resnet_Model.Parser_args import parse_Arg
from Utils import get_device
from Utils import CIFARLoadData

args = parse_Arg()
train_loader = CIFARLoadData(args.batch_size, True, False)
test_loader = CIFARLoadData(args.batch_size, False, False)

device = get_device()

# 50 Resnet
model = ResNet(args.channels, args.layers).to(device)

# 학습 진행
for epoch in range(args.n_epochs):
    for i, data in enumerate(train_loader):
        inputs, labels = data[0].to(device), data[1].to(device)

        loss = model.Learn(inputs, labels)

        print("[Epoch %d/%d] [Batch %d/%d] [loss: %f]]" % (epoch, args.n_epochs, i, len(train_loader), loss))

# 평가
with torch.no_grad():
    model.eval()

    X_test = test_loader.test_data.view(len(test_loader), 1, 28, 28).float()
    Y_test = test_loader.test_labels

    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())

