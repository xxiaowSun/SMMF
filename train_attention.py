import os
import copy
import torch
import numpy as np

from opt import OptInit
from utils.tools import (EarlyStopping, feature_selection, print_result, save_result)
from utils.metrics import accuracy, auc, metrics
from utils.mydataloader import MyDataloader
from model.mm_attention import MM_GTUNets
from model.rp_graph import create_reward_penalty_graph
from tensorboardX import SummaryWriter


def rampup(epoch, start, end):
    if epoch < start:
        return 0.0
    if epoch > end:
        return 1.0
    return float(epoch - start) / float(end - start)


def build_class_weight(y, train_ind, device):
    cls_count = np.bincount(y[train_ind])
    cls_count = np.maximum(cls_count, 1)
    w = cls_count.sum() / cls_count
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32, device=device)


class MetricSmoother:
    """对 val_auc 做 EMA 平滑，减少某些 fold 的验证抖动"""
    def __init__(self, beta=0.8):
        self.beta = beta
        self.val = None

    def update(self, x):
        x = float(x)
        if self.val is None:
            self.val = x
        else:
            self.val = self.beta * self.val + (1 - self.beta) * x
        return self.val


# =========================
# ✅ 手动 SWA（避免 deepcopy）
# =========================
def make_swa_state_dict(model):
    """创建 SWA state：只存参数的 detach clone（叶子张量），避免 deepcopy(model)"""
    swa_sd = {}
    for k, v in model.state_dict().items():
        if torch.is_tensor(v):
            swa_sd[k] = v.detach().clone()
        else:
            swa_sd[k] = v
    return swa_sd


@torch.no_grad()
def update_swa_state_dict(swa_sd, model_sd, num_averaged):
    """
    swa = (swa * num + cur) / (num + 1)
    只对浮点张量做平均；整型/bool（如 num_batches_tracked）直接覆盖
    """
    for k, v in model_sd.items():
        if k not in swa_sd:
            continue
        if torch.is_tensor(v) and torch.is_tensor(swa_sd[k]):
            if v.is_floating_point():
                swa_sd[k].mul_(num_averaged / (num_averaged + 1.0)).add_(
                    v.detach(), alpha=1.0 / (num_averaged + 1.0)
                )
            else:
                swa_sd[k] = v.detach().clone()
        else:
            swa_sd[k] = v


@torch.no_grad()
def refresh_bn_by_full_forward(model, x, ph_features, affinity_graphs, iters=10):
    """
    你是“全图 forward”的训练方式，没有 DataLoader 去 update_bn，
    这里用多次 forward 刷新 BN running stats。
    """
    model.train()
    for _ in range(iters):
        _ = model(x, ph_features, affinity_graphs)
    model.eval()


def train_one_run(run_id=0, use_swa=True):
    """
    单次训练（某些 fold 可重复训练取最好）
    返回：best_state_dict, best_val_auc（平滑后的best）, best_val_acc
    """
    print(f"  [Run {run_id}] Start training...")

    best_val_auc = -1.0
    best_val_acc = 0.0
    best_state = None
    best_epo = 0

    # warmup 区间：前 20% 不开 aux，20%-60% 逐步打开
    start_ep = int(0.2 * opt.epoch)
    end_ep = int(0.6 * opt.epoch)

    # val_auc 平滑用于 early-stop/save
    smoother = MetricSmoother(beta=getattr(opt, "val_smooth_beta", 0.8))

    # SWA：后 30% epoch 开始
    swa_start = int(0.7 * opt.epoch)
    swa_sd = None
    num_averaged = 0

    for epoch in range(opt.epoch):
        alpha = rampup(epoch, start_ep, end_ep)

        # -------- Train --------
        model.train()
        optimizer.zero_grad()

        outputs, aux_loss, loss_dict = model(x, ph_features, affinity_graphs)
        out_tr = outputs[train_ind]
        ce_loss = loss_fn(out_tr, labels[train_ind])
        loss = ce_loss + alpha * aux_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=getattr(opt, "clip_grad", 1.0))
        optimizer.step()

        # cosine scheduler
        if base_scheduler is not None:
            base_scheduler.step()

        # -------- Val --------
        model.eval()
        with torch.no_grad():
            outputs, aux_loss_val, _ = model(x, ph_features, affinity_graphs)
            out_val = outputs[val_ind]
            ce_val = loss_fn(out_val, labels[val_ind])

        out_val_np = out_val.detach().cpu().numpy()
        _, acc_val = accuracy(out_val_np, y[val_ind])
        val_auc_raw = auc(out_val_np, y[val_ind])

        # 平滑 auc 用于保存/earlystop
        val_auc_s = smoother.update(val_auc_raw)

        # -------- Logging --------
        if opt.log_save:
            writer.add_scalar(f"run{run_id}/train/loss_total", loss.item(), epoch)
            writer.add_scalar(f"run{run_id}/train/loss_ce", ce_loss.item(), epoch)
            writer.add_scalar(f"run{run_id}/train/loss_aux", aux_loss.item(), epoch)
            writer.add_scalar(f"run{run_id}/train/alpha", alpha, epoch)

            writer.add_scalar(f"run{run_id}/val/auc_raw", float(val_auc_raw), epoch)
            writer.add_scalar(f"run{run_id}/val/auc_smooth", float(val_auc_s), epoch)
            writer.add_scalar(f"run{run_id}/val/acc", float(acc_val), epoch)
            writer.add_scalar(f"run{run_id}/val/loss_ce", ce_val.item(), epoch)
            writer.add_scalar(f"run{run_id}/val/loss_aux", aux_loss_val.item(), epoch)

        if epoch % opt.print_freq == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  [Run {run_id}] Epoch {epoch:03d} | lr {lr:.6f} | alpha {alpha:.3f} | "
                f"train_ce {ce_loss.item():.4f} aux {aux_loss.item():.4f} | "
                f"val_auc {val_auc_raw:.4f} (smooth {val_auc_s:.4f}) | val_acc {float(acc_val):.4f}"
            )

        # -------- Save best by smoothed AUC --------
        if val_auc_s > best_val_auc + getattr(opt, "min_delta_auc", 1e-4):
            best_val_auc = val_auc_s
            best_val_acc = float(acc_val)
            best_epo = epoch
            best_state = copy.deepcopy(model.state_dict())

        # -------- Manual SWA update (no deepcopy(model)) --------
        if use_swa and epoch >= swa_start:
            if swa_sd is None:
                swa_sd = make_swa_state_dict(model)
                num_averaged = 1
            else:
                update_swa_state_dict(swa_sd, model.state_dict(), num_averaged)
                num_averaged += 1

        # -------- Early stop by smoothed AUC (convert to "loss") --------
        score = -best_val_auc  # 越小越好
        early_stopping(score, model)
        if early_stopping.early_stop:
            print(f"  [Run {run_id}] Early stopping at epoch {epoch}, best epoch {best_epo}")
            break

    # 训练结束：如果启用 SWA，则用 SWA 权重替换 best_state（对最差 fold 通常更稳）
    if use_swa and swa_sd is not None:
        model.load_state_dict(swa_sd, strict=False)
        refresh_bn_by_full_forward(model, x, ph_features, affinity_graphs, iters=10)
        best_state = copy.deepcopy(model.state_dict())
        print(f"  [Run {run_id}] Manual SWA applied (BN refreshed by repeated forward).")

    return best_state, best_val_auc, best_val_acc


def evaluate_with_state(state_dict, tag="best"):
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    with torch.no_grad():
        outputs, aux_loss, _ = model(x, ph_features, affinity_graphs)

    outputs_test = outputs[test_ind].detach().cpu().numpy()
    correct, acc = accuracy(outputs_test, y[test_ind])
    sen, spe, f1v = metrics(outputs_test, y[test_ind])
    aucv = auc(outputs_test, y[test_ind])

    print(f"  [{tag}] Fold {fold} test acc {acc:.5f}, AUC {aucv:.5f}, aux {float(aux_loss.detach().cpu()):.5f}")
    return correct, acc, sen, spe, f1v, aucv


if __name__ == "__main__":
    settings = OptInit(model="MM_GTUNets", dataset="ADHD", atlas="aal")
    settings.args.device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")

    # ---------- 稳健默认（即使 opt 里没配也能跑） ----------
    settings.args.w_mod = getattr(settings.args, "w_mod", 0.2) #0.2
    settings.args.w_tok = getattr(settings.args, "w_tok", 0.01)
    settings.args.w_fus = getattr(settings.args, "w_fus", 0.2)
    settings.args.w_graph = getattr(settings.args, "w_graph", 1.0)
    settings.args.clip_grad = getattr(settings.args, "clip_grad", 1.0)
    settings.args.use_class_weight = getattr(settings.args, "use_class_weight", 1)

    # 训练稳定性参数
    settings.args.min_delta_auc = getattr(settings.args, "min_delta_auc", 1e-4)
    settings.args.val_smooth_beta = getattr(settings.args, "val_smooth_beta", 0.8)

    # hard folds：第1折和第10折（fold=0,9）重复训练取最好
    settings.args.hard_folds = getattr(settings.args, "hard_folds", [0, 9])
    settings.args.hard_fold_repeats = getattr(settings.args, "hard_fold_repeats", 2)
    # --------------------------------------------------------

    opt = settings.initialize()
    settings.print_args()

    dl = MyDataloader(opt)
    x, y, ph_dict, ph_data = dl.load_data(save=False)

    labels = torch.tensor(y, dtype=torch.long).to(opt.device)
    ph_features = torch.from_numpy(ph_data).float()

    n_folds = opt.folds
    cv_splits = dl.data_split(n_folds, val_ratio=0.1)

    corrects = np.zeros(n_folds, dtype=np.int32)
    accs = np.zeros(n_folds, dtype=np.float32)
    sens = np.zeros(n_folds, dtype=np.float32)
    spes = np.zeros(n_folds, dtype=np.float32)
    f1 = np.zeros(n_folds, dtype=np.float32)
    aucs = np.zeros(n_folds, dtype=np.float32)

    for fold in range(n_folds):
        print("\n========================== Fold {} ==========================".format(fold))
        train_ind = cv_splits[fold][0]
        val_ind = cv_splits[fold][1]
        test_ind = cv_splits[fold][2]

        # feature_selection
        x = feature_selection(x, y, train_ind, opt.node_dim)

        affinity_graphs = torch.from_numpy(
            create_reward_penalty_graph(ph_dict, y, train_ind, val_ind, test_ind, opt)
        ).float()

        model = MM_GTUNets(opt, fold).to(opt.device)
        print(model)

        # loss_fn：类别不平衡处理
        if getattr(opt, "use_class_weight", 1) == 1:
            class_w = build_class_weight(y, train_ind, opt.device)
            loss_fn = torch.nn.CrossEntropyLoss(weight=class_w)
        else:
            loss_fn = torch.nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.wd)
        base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epoch)

        early_stopping = EarlyStopping(patience=opt.early_stop, verbose=True)

        if opt.log_save:
            writer = SummaryWriter(f"./log/{opt.model}_{opt.dataset}_{opt.atlas}_log/{fold}")

        fold_model_path = opt.ckpt_path + "/fold{}.pth".format(fold)

        # -------- 训练：hard fold 重复训练取最好 --------
        if opt.train == 1:
            # pretrain vae（保留）
            model.train_vae(ph_features)

            repeats = opt.hard_fold_repeats if fold in opt.hard_folds else 1
            best_state = None
            best_auc_s = -1.0

            for run_id in range(repeats):
                # 每次 run 重新初始化优化器/调度器/earlystop，避免被上次 run 影响
                optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.wd)
                base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epoch)
                early_stopping = EarlyStopping(patience=opt.early_stop, verbose=True)

                state, auc_s, acc_s = train_one_run(run_id=run_id, use_swa=True)
                if state is not None and auc_s > best_auc_s:
                    best_auc_s = auc_s
                    best_state = state

            # 保存 best_state
            if (opt.ckpt_path != '') and opt.model_save and best_state is not None:
                if not os.path.exists(opt.ckpt_path):
                    os.makedirs(opt.ckpt_path)
                torch.save(best_state, fold_model_path)
                print(f"Saved BEST model for fold {fold} to {fold_model_path}")

            # 测试：用 best_state
            if best_state is None:
                best_state = torch.load(fold_model_path, map_location=opt.device)

            correct, acc, sen, spe, f1v, aucv = evaluate_with_state(best_state, tag="best")

        else:
            # 只测
            state = torch.load(fold_model_path, map_location=opt.device)
            correct, acc, sen, spe, f1v, aucv = evaluate_with_state(state, tag="loaded")

        corrects[fold] = correct
        accs[fold] = acc
        sens[fold] = sen
        spes[fold] = spe
        f1[fold] = f1v
        aucs[fold] = aucv

    print("\n========================== Finish ==========================")
    print_result(opt, n_folds, accs, sens, spes, aucs, f1)
    save_result(opt, n_folds, accs, sens, spes, aucs, f1)
