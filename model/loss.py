import torch
import torch.nn.functional as F


def split_polygon_pred(pred, num_classes):
    obj = pred[:, 0:1]
    cls = pred[:, 1:1 + num_classes]
    points = pred[:, 1 + num_classes:1 + num_classes + 8]
    return obj, cls, points


def build_targets_batch(labels_batch, grid_h, grid_w, num_classes, device):
    out_dim = 1 + num_classes + 8
    batch_size = len(labels_batch)

    target = torch.zeros(batch_size, out_dim, grid_h, grid_w, device=device)
    obj_mask = torch.zeros(batch_size, 1, grid_h, grid_w, device=device)

    for b, labels in enumerate(labels_batch):
        labels = labels.to(device)

        for label in labels:
            cls_id = int(label[0].item())
            points = label[1:9]

            xs = points[0::2]
            ys = points[1::2]

            cx = xs.mean()
            cy = ys.mean()

            gx = min(int(cx * grid_w), grid_w - 1)
            gy = min(int(cy * grid_h), grid_h - 1)

            target[b, 0, gy, gx] = 1.0
            target[b, 1 + cls_id, gy, gx] = 1.0
            target[b, 1 + num_classes:1 + num_classes + 8, gy, gx] = points

            obj_mask[b, 0, gy, gx] = 1.0

    return target, obj_mask


def polygon_loss(pred, labels_batch, num_classes):
    device = pred.device
    batch_size, _, grid_h, grid_w = pred.shape

    target, obj_mask = build_targets_batch(
        labels_batch=labels_batch,
        grid_h=grid_h,
        grid_w=grid_w,
        num_classes=num_classes,
        device=device,
    )

    pred_obj, pred_cls, pred_points = split_polygon_pred(pred, num_classes)
    tgt_obj, tgt_cls, tgt_points = split_polygon_pred(target, num_classes)

    obj_loss = F.binary_cross_entropy_with_logits(pred_obj, tgt_obj)

    cls_loss = F.binary_cross_entropy_with_logits(pred_cls, tgt_cls)

    pred_points = pred_points.sigmoid()

    point_loss = F.smooth_l1_loss(
        pred_points * obj_mask,
        tgt_points * obj_mask,
    )

    return obj_loss + cls_loss + 5.0 * point_loss