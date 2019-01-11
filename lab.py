import torch

def accu_iou(pred_y, y):
        # B is the mask pred, A is the malanoma 
    y_pred = ( pred_y > 0.7) * 1.0
    y_true = ( y > 0.5) * 1.0
    pred_flat = y_pred.view(y_pred.numel())
    true_flat = y_true.view(y_true.numel())

    intersection = (torch.sum(pred_flat * true_flat)) + 1
    # torch.set_default_tensor_type(torch.FloatType)
    denominator = (torch.sum(pred_flat + true_flat)) - intersection + 100

    matrix_iou = (intersection/denominator)
    return  matrix_iou

pred_y = torch.Tensor([1,0,0,1])
y = torch.Tensor([1,1,1,0])

iou = accu_iou(pred_y, y)


re = a/b


print(iou)