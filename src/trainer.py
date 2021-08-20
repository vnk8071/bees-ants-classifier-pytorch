from matplotlib import cm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter

from src.dataset import BeeAntDataset
from src.model import ResNet
import os
import time


writer = SummaryWriter("runs/bees-ants-classifier")


class Trainer(object):
    def __init__(self, **args):
        for key in args:
            setattr(self, key.upper(), args[key])


class BeeAntClassifier(Trainer):
    def __init__(self, **args):
        super(BeeAntClassifier, self).__init__(**args)

    def set_up_training(self):
        model = ResNet(self.NUM_CLASSES)
        loss_criteria = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=self.LEARNING_RATE)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return model, optimizer, loss_criteria, device

    def train_model(self):
        since = time.time()
        [model, optimizer, loss_criteria, device] = self.set_up_training()
        NUM_EPOCHS = self.MAX_EPOCHS
        dataset_sizes, dataloaders = BeeAntDataset(
            self.DATA_DIR, self.IMAGE_SIZE).set_up_training_data(self.BATCH_SIZE, self.NUM_WORKERS)

        examples = iter(dataloaders['train'])
        image, _ = examples.__next__()
        img_grid = torchvision.utils.make_grid(image, normalize=True)

        writer.add_image('bee_ants_images', img_grid)
        writer.add_graph(model, image)

        best_acc = 0.0
        best_val_loss = float('inf')
        nonimproved_epoch = 0

        # Save model
        if not os.path.exists('./models'):
            os.makedirs('./models')

        for epoch in range(NUM_EPOCHS):
            print('Epoch {}/{}'.format(epoch+1, NUM_EPOCHS))
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                for images, labels in dataloaders[phase]:
                    images = images.to(device)
                    labels = labels.to(device)

                    # forward
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(images)
                        _, preds = torch.max(outputs, 1)
                        loss = loss_criteria(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * images.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / \
                    dataset_sizes[phase]
                print('[{}] - Loss: {:.4f} - Acc: {:.4f}'.format(
                    phase.upper(), epoch_loss, epoch_acc))

                if phase == 'train':
                    writer.add_scalar('train loss', epoch_loss, epoch+1)
                    writer.add_scalar('train accuracy', epoch_acc, epoch+1)
                else:
                    writer.add_scalar('val loss', epoch_loss, epoch+1)
                    writer.add_scalar('val accuracy', epoch_acc, epoch+1)
                running_loss = 0.0
                running_corrects = 0
                writer.close()

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save({'model': model.state_dict(),
                                'best_accuracy': best_acc,
                                'classes': ['bee', 'ant'],
                                'optimizer': optimizer.state_dict()}, 'models/beeant_epoch{}_accuracy{:.4f}.pth'.format(epoch, epoch_acc))
                elif phase == 'val' and epoch_acc < best_acc:
                    nonimproved_epoch += 1

                if nonimproved_epoch > 5:
                    print('Early stopping. Model not improving.')
                    break

                if phase == 'val' and epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    torch.save({'model': model.state_dict(),
                                'loss': loss,
                                'classes': ['bee', 'ant'],
                                'optimizer': optimizer.state_dict()}, self.SAVE_MODEL_PATH)
            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))
        print('TRAINING DONE')
