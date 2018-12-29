import torch, time, os, torch.nn as nn
import torch.optim as optim
from models import DCOD300_Body, MultiBoxLoss
from datasets import ListDataset
import warnings
from utils import progress_bar
warnings.filterwarnings('ignore')

# hyparametres
batch_size = 96
n_cpu = 32
gpu = list(range(6))
train_path = 'train.txt'
multi_gpu = True
mode = 'train'
epochs = 100
check_dir = 'checkpoints'
interval = 10
classes = 21

# file operation
os.makedirs(check_dir, exist_ok=True)

# model
model = DCOD300_Body(growth_rate=32)
if multi_gpu:
    model = nn.DataParallel(model, device_ids=gpu).cuda()
else:
    model = model.cuda()
if 'backup.pkl' in os.listdir(check_dir):
    print('Load existed model!')
    model.load_state_dict(torch.load(os.path.join(check_dir, 'backup.pkl')))

# datasets
dataloader = torch.utils.data.DataLoader(
    ListDataset(train_path), batch_size=batch_size, shuffle=True, num_workers=n_cpu
)

# optimizer and loss function
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
criterion = MultiBoxLoss(classes, .5, True, 3, .5)

if mode == 'train':
    seen, start = 0, time.time()
    for epoch in range(epochs):
        for batch_i, (imgs, targets) in enumerate(dataloader):
            imgs = imgs.type(torch.cuda.FloatTensor)
            targets = targets.type(torch.cuda.FloatTensor)
            optimizer.zero_grad()
            output = model(imgs)
            loss_l, loss_c = criterion(output, targets)
            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()
            seen += targets.size(0)
            runtime = time.time() - start
            progress_bar(batch_i, len(dataloader), 'epoch %d, loss %.2f, speed %.2f' % (epoch, loss.item(), seen / runtime))

        if epoch % interval == 0:
            torch.save(model.state_dict(), '%s/%s.pkl' % (check_dir, epoch))
            torch.save(model.state_dict(), '%s/backup.pkl' % check_dir)
