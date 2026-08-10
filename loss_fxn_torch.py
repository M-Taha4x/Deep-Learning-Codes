import torch
import torch.nn as nn
bce_loss_fn=nn.BCELoss()
predicted_prob=torch.tensor([0.9]) #y_predicted
true_label=torch.tensor([1.0]) #y_actual
loss=bce_loss_fn(predicted_prob,true_label)
print(f"BCE loss(Good Prediction):{loss.item():.4f}")

cce_loss_fn=nn.CrossEntropyLoss()
predicted_scores=torch.tensor([[2.5,-1.2,0.3]])
true_class=torch.tensor([0])
loss2=cce_loss_fn(predicted_scores,true_class)
print(f"CCE Loss: {loss2.item():.4f}")
