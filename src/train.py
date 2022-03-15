import torch
import torch.nn as nn
from model_v1 import EmotionClassifier
from dataloader import TRAIN_LOADER

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_model():
    model = EmotionClassifier()

    criterion = nn.CrossEntropyLoss()

    loss_fn = criterion

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    return model.to(device), loss_fn, optimizer


model, criterion, optimizer = get_model()


def train_batch(data, model, optimizer, criteria):
    model.train()
    imgs, labels = data

    imgs = imgs.to(device)
    labels = labels.type(torch.LongTensor)
    labels = labels.to(device)

    optimizer.zero_grad()
    pred_label = model(imgs)

    criterion = criteria(pred_label, labels)

    criterion.backward()
    optimizer.step()

    return criterion


def validate_batch(data, model, criteria):
    model.eval()

    imgs, labels = data

    imgs = imgs.to(device)
    labels = labels.type(torch.LongTensor)
    labels = labels.to(device)

    with torch.no_grad():
        pred_label = model(imgs)

    criterion = criteria(pred_label, labels.type(torch.LongTensor))

    return criterion



if __name__ == "__main__":
    import time
    from tqdm import tqdm
    from tqdm import trange

    model, criteria, optimizer = get_model()

    train_losses = []

    val_losses = []
    n_epochs = 100
    best_test_loss = 1000
    start = time.time()

    for epoch in trange(n_epochs, desc="Epoch : "):

        epoch_train_loss, epoch_test_loss = 0, 0
        ctr = 0
        _n = len(TRAIN_LOADER)

        for ix, data in enumerate(tqdm(TRAIN_LOADER, position=0, leave=True)):
            loss = train_batch(data, model, optimizer, criteria)
            epoch_train_loss += loss.item()
        
        # for ix, data in enumerate(tqdm(test_loader, position=0, leave=True)):
        #     loss, gender_acc, age_mae = validate_batch(data, model, criteria)
        #     epoch_test_loss += loss.item()
        #     val_age_mae += age_mae
        #     val_gender_acc += gender_acc
        #     ctr += len(data[0])
        
        epoch_train_loss /= len(TRAIN_LOADER)
        # epoch_test_loss /= len(test_loader)


        elapsed = time.time()-start
        best_test_loss = min(best_test_loss, epoch_test_loss)
        print('{}/{} ({:.2f}s - {:.2f}s remaining)'.format(\
            epoch+1, n_epochs, time.time()-start, \
            (n_epochs-epoch)*(elapsed/(epoch+1)))
            )

        info = f'''Epoch: {epoch+1:03d}\
        \tTrain Loss: {epoch_train_loss:.3f}
        \tTest:{epoch_test_loss:.3f}
        \tBest Test Loss: {best_test_loss:.4f}'''

        print(info)

    torch.save(model.state_dict(), 'src/models/model.pth')
