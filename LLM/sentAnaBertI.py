from torch import nn
from models.bert_model import *


"""使用mean max pool的方式進行情感分析"""

class Bert_Sentiment_Analysis(nn.Module):
    def __init__(self, config):
        super(Bert_Sentiment_Analysis, self).__init__()
        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.final_dense = nn.Linear(config.hidden_size, 1)
        self.activation = nn.Sigmoid()

    def compute_loss(self, predictions, labels):
        # 將預測和標記的維度展平, 防止出現維度不一致
        predictions = predictions.view(-1)
        labels = labels.float().view(-1)
        epsilon = 1e-8
        loss =-labels * torch.log(predictions + epsilon) - (torch.tensor(1.0) - labels) * torch.log(torch.tensor(1.0) - predictions + epsilon)  # 交叉熵
        # 求均值, 并返回可以反傳的loss
        # loss爲一個實數
        loss = torch.mean(loss)
        return loss

    def forward(self, text_input, positional_enc, labels=None):
        encoded_layers, _ = self.bert(text_input, positional_enc,
                                    output_all_encoded_layers=True)
        sequence_output = encoded_layers[2]
        # # sequence_output的維度是[batch_size, seq_len, embed_dim]
        avg_pooled = sequence_output.mean(1)
        max_pooled = torch.max(sequence_output, dim=1)
        pooled = torch.cat((avg_pooled, max_pooled[0]), dim=1)
        pooled = self.dense(pooled)



        # 下面是[batch_size, hidden_dim * 2] 到 [batch_size, 1]的映射
        # 我們在這裏要解決的是二分類問題

        predictions = self.final_dense(pooled)

        # 用sigmoid函數做激活, 返回0-1之間的值
        predictions = self.activation(predictions)
        if labels is not None:
            # 計算loss
            loss = self.compute_loss(predictions, labels)
            return predictions, loss
        else:
            return predictions
