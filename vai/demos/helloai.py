def main():
    if __name__ == 'main':
        raise RuntimeError('This script should only be called from a Jupyter Notebook')

    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    from tqdm import tqdm_notebook as tqdm

    from vai.images import show_images
    from vai.plot import smooth_plot

    # PyTorch Modules
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision.transforms as transforms
    from torch.autograd import Variable
    from torch.utils.data.dataloader import DataLoader
    from torch.utils.data.sampler import SubsetRandomSampler
    from torchvision.datasets import MNIST

    batch_size = 64
    val_split = 0.2

    # Define Useful Features
    dir_data = Path('~/.data/MNIST').expanduser()
    dir_data.mkdir(parents=True, exist_ok=True)

    def _flatten(x):
        return x.view(x.size(0), -1)

    def test_model():
        model.eval()
        images = torch.zeros(5, 10, 28, 28)

        images_gethered = [0] * 10
        for x, _ in data_val:
            x = Variable(x, volatile=True).squeeze(1)
            y = model(_flatten(x)).max(1)[1].data
            x = x.data

            for i in range(10):
                if images_gethered[i] >= 5:
                    continue

                y_i = torch.arange(0, len(y))[y == i].long()
                if len(y_i) == 0:
                    continue

                x_i = x.index_select(0, y_i)
                x_i = x_i[:min(len(x_i), 5 - images_gethered[i])]
                images[images_gethered[i]:images_gethered[i] + len(x_i), i] = x_i

                images_gethered[i] += len(x_i)

            if all(i >= 5 for i in images_gethered):
                break

        show_images(images.view(-1, 28, 28).cpu().numpy(),
                    cmap='gray', pixel_range=(-1, 1))

        model.train()

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.5] * 3, [0.5] * 3)])

    _dataset = MNIST(dir_data, transform=transform, download=True)
    _len_train = int((1 - val_split) * len(_dataset))

    data = DataLoader(_dataset, batch_size,
                      sampler=SubsetRandomSampler(list(range(_len_train))))
    data_val = DataLoader(_dataset, batch_size,
                          shuffle=False, drop_last=True,
                          sampler=SubsetRandomSampler(list(range(_len_train, len(_dataset)))))

    def data_generator():
        while True:
            data_iterator = iter(data)
            for x, y in data_iterator:
                yield Variable(_flatten(x)), Variable(y)

    model = nn.Linear(784, 10)
    history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}
    optimizer = optim.Adam(model.parameters())

    criterion = nn.CrossEntropyLoss()

    def _get_metrics(y_pred, y):
        return criterion(y_pred, y), (y_pred.max(1)[1] == y).float().mean().data.cpu().numpy()[0]

    def _append_val():
        model.eval()

        x, y = next(iter(data_val))
        x = Variable(x, volatile=True)
        y = Variable(y, volatile=True)
        y_pred = model(_flatten(x))

        loss, acc = _get_metrics(y_pred, y)

        history['val_loss'].append(loss.data.cpu().numpy()[0])
        history['val_acc'].append(acc)

        model.train()

    def optimize(epochs=1):
        iterations = int(len(data) * epochs)
        gen = data_generator()

        for batch in tqdm(range(iterations)):
            x, y = next(gen)
            y_pred = model(x)

            loss, acc = _get_metrics(y_pred, y)
            loss.backward()
            optimizer.step()

            optimizer.zero_grad()

            history['loss'].append(loss.data.cpu().numpy()[0])
            history['acc'].append(acc)
            if batch % 50 == 0:
                _append_val()

    print("Here's the model before optimization")
    test_model()

    print('Optimizing now. Hang on tight!')
    optimize(1)

    # Show Results

    smooth_plot(history['loss'], label='training', replace_outliers=False)
    plt.plot(np.arange(len(history['val_loss'])) * 50,
             history['val_loss'], label='validation')
    plt.legend()
    plt.title('Loss')
    plt.show()

    smooth_plot(history['acc'], label='training', replace_outliers=False)
    plt.plot(np.arange(len(history['val_acc'])) * 50,
             history['val_acc'], label='validation')
    plt.legend()
    plt.title('Accuracy')
    plt.show()

    print("Here's the optimized model")
    test_model()


if __name__ != 'main':
    main()
