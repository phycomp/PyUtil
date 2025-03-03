from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset as 映射dset
# 超參數
EPOCHS, LR, 束量=10, 5, 64# epoch, learning rate, batch size for training
from torch.nn import CrossEntropyLoss as 交叉熵
#交叉熵 = CrossEntropyLoss()
from torch.optim import SGD     #as 最佳化 optimizer
最佳化 = SGD(model.parameters(), lr=LR)
from torch.optim.lr_scheduler import StepLR
scheduler = StepLR(最佳化, 1., gamma=.1)
精確度 = None
train_iter, test_iter = AG_NEWS()
訓練集 = 映射dset(train_iter)
測試集 = 映射dset(test_iter)
num_train = int(len(訓練集) * 0.95)
split_train_, split_valid_ = random_split(訓練集, [num_train, len(訓練集) - num_train])

train_dataloader = DataLoader(split_train_, batch_size=束量, shuffle=True, collate_fn=collate_batch)
valid_dataloader = DataLoader(split_valid_, batch_size=束量, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(測試集, batch_size=束量, shuffle=True, collate_fn=collate_batch)

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(train_dataloader)
    accu_val = evaluate(valid_dataloader)
    #if 精確度 is not None and 
    if 精確度 > accu_val:
      scheduler.step()
    else:
       精確度 = accu_val
    print('-' * 59)
    print('| end of epoch {:3d} | time: {:5.2f}s | '
          'valid accuracy {:8.3f} '.format(epoch, time.time() - epoch_start_time, accu_val))
    print('-' * 59)
