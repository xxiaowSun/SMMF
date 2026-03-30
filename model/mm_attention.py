import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import dropout_edge, dense_to_sparse

from utils.tools import cal_feature_sim, EarlyStopping
from model.gtunet import GTUNet


class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()
        self.args = args
        self.input_dim = len(args.scores)
        self.hidden_dim = 256
        self.latent_dim = args.node_dim
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.latent_dim * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_dim),
            nn.Sigmoid()
        )
        self.criterion = nn.MSELoss(reduction='sum')

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu_logvar = self.encoder(x)
        mu = mu_logvar[:, :self.latent_dim]
        logvar = mu_logvar[:, self.latent_dim:]
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        reconstruction_loss = self.criterion(recon_x, x)
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = reconstruction_loss + kl_divergence
        return z, total_loss


# -------------------------
# (2) RP_Attention: �������ʵ��
# -------------------------
class CustomConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels):
        super(CustomConv2d, self).__init__(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.weight.data = torch.abs(self.weight.data)
        self.weight.data[:, 1, :, :] = - self.weight.data[:, 1, :, :]

    def forward(self, x):
        self.weight.data[:, 1, :, :] = -torch.max(torch.abs(self.weight.data[:, 1, :, :]),
                                                  self.weight.data[:, 1, :, :] + self.weight.data[:, 2, :, :])
        self.weight.data[:, 0, :, :] = torch.max(self.weight.data[:, 0, :, :], self.weight.data[:, 2, :, :])
        return super(CustomConv2d, self).forward(x)


class RP_Attention(nn.Module):
    def __init__(self, args):
        super(RP_Attention, self).__init__()
        self.args = args
        self.num_score = len(args.scores)
        self.device = args.device

        self.weights = nn.Parameter(torch.rand(self.num_score)).to(self.device)
        self.weights.data /= self.weights.data.sum()

        self.conv = CustomConv2d(in_channels=3, out_channels=1)
        self.convs = torch.nn.ModuleList()
        for _ in range(self.num_score):
            self.convs.append(self.conv)

    def forward(self, affinity_graphs):
        outputs = []
        value = []
        for i in range(self.num_score):
            outputs.append(torch.sigmoid(self.convs[i](affinity_graphs[i]).squeeze()))
            reward = self.convs[i].weight.data[:, 0, :, :].squeeze()
            penalty = self.convs[i].weight.data[:, 1, :, :].squeeze()
            value.append(self.cal_value(affinity_graphs[i], reward, penalty))

        outputs = torch.stack(outputs)
        rp_graph = (self.weights.view(self.num_score, 1, 1) * outputs).sum(dim=0)
        value = torch.tensor(value, device=rp_graph.device).sum()
        return rp_graph, value

    def cal_value(self, graphs, reward, penalty):
        reward_graph = graphs[0, :, :]
        penalty_graph = graphs[1, :, :]
        value = torch.sum(F.relu(reward * reward_graph + penalty * penalty_graph)) / (self.args.num_subjects ** 2)
        return value


# -------------------------
# (3) Multimodal_Attention: �������ʵ��
# -------------------------
class Multimodal_Attention(nn.Module):
    def __init__(self, args):
        super(Multimodal_Attention, self).__init__()
        self.channel = args.out
        self.shared_lin = nn.Linear(self.channel, self.channel)
        self.img_lin = nn.Linear(self.channel, self.channel)
        self.ph_lin = nn.Linear(self.channel, self.channel)

    def cal_attention_score(self, attention, shared_attention):
        denom = torch.trace(torch.mm(shared_attention, shared_attention.t())) + 1e-8
        return torch.trace(torch.mm(attention, attention.t())) / denom

    def forward(self, img_embed, ph_embed):
        shared_embed = 0.5 * (img_embed + ph_embed)

        img_attention = torch.tanh(self.img_lin(img_embed))
        ph_attention = torch.tanh(self.ph_lin(ph_embed))
        shared_attention = torch.tanh(self.shared_lin(shared_embed))

        img_shared_score = self.cal_attention_score(img_attention, shared_attention)
        ph_shared_score = self.cal_attention_score(ph_attention, shared_attention)
        attention_scores = F.softmax(torch.tensor([img_shared_score, ph_shared_score], device=img_embed.device), dim=0)

        img_weight = attention_scores[0]
        ph_weight = attention_scores[1]

        joint_embed = shared_attention * shared_embed + img_attention * img_embed + ph_attention * ph_embed
        return joint_embed, img_weight, ph_weight


# ============================================================
# ����ģ�飨3.1 / 3.2 / 3.3��������֮ǰ�汾һ�� + �Ƚ��Ķ�
# ============================================================
class GaussianUncertaintyHead(nn.Module):
    def __init__(self, in_dim, latent_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, latent_dim * 2)
        )
        self.latent_dim = latent_dim

    def forward(self, g):
        h = self.proj(g)
        mu = h[:, :self.latent_dim]
        logvar = h[:, self.latent_dim:]
        return mu, logvar


def symmetric_kl_diag_gaussian(mu1, logvar1, mu2, logvar2):
    var1 = torch.exp(logvar1)
    var2 = torch.exp(logvar2)

    kl12 = 0.5 * torch.sum(
        (var1 / (var2 + 1e-8)) +
        ((mu2 - mu1) ** 2) / (var2 + 1e-8) +
        (logvar2 - logvar1) - 1.0,
        dim=1
    )
    kl21 = 0.5 * torch.sum(
        (var2 / (var1 + 1e-8)) +
        ((mu1 - mu2) ** 2) / (var1 + 1e-8) +
        (logvar1 - logvar2) - 1.0,
        dim=1
    )
    return (kl12 + kl21).mean()


class Tokenizer(nn.Module):
    def __init__(self, in_dim, num_tokens, token_dim):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.proj = nn.Linear(in_dim, num_tokens * token_dim)

    def forward(self, x):
        z = self.proj(x)
        return z.view(x.size(0), self.num_tokens, self.token_dim)


def ridge_reconstruction_operator(X_src, X_tgt, lam=1e-2):
    # X: (T, d)
    d = X_src.size(1)
    A = X_src.t().mm(X_src) + lam * torch.eye(d, device=X_src.device, dtype=X_src.dtype)
    B = X_tgt.t().mm(X_src)
    W = torch.linalg.solve(A.t(), B.t()).t()
    return W


def token_bidir_operator_loss(tokens_a, tokens_b, lam=1e-2):
    device = tokens_a.device
    N, T, d = tokens_a.shape
    I = torch.eye(d, device=device, dtype=tokens_a.dtype)

    losses = []
    for i in range(N):
        Xa = tokens_a[i]
        Xb = tokens_b[i]
        W_ab = ridge_reconstruction_operator(Xa, Xb, lam=lam)
        W_ba = ridge_reconstruction_operator(Xb, Xa, lam=lam)
        M = W_ab.mm(W_ba)
        losses.append(((M - I) ** 2).mean())
    return torch.stack(losses).mean()


class SignalNoiseDecomposer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        s = self.mlp(x)
        n = x - s
        return s, n


class TwoViewFusion(nn.Module):
    def __init__(self, dim, temp=0.07):
        super().__init__()
        self.decomp_img = SignalNoiseDecomposer(dim)
        self.decomp_txt = SignalNoiseDecomposer(dim)
        self.temp = temp

    def info_nce(self, q, k):
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        logits = q @ k.t() / self.temp
        labels = torch.arange(q.size(0), device=q.device)
        return F.cross_entropy(logits, labels)

    def forward(self, img_feat, txt_feat):
        s_img, n_img = self.decomp_img(img_feat)
        s_txt, n_txt = self.decomp_txt(txt_feat)

        # multi-view consistency
        L_mv_a = self.info_nce(s_txt, img_feat) + s_txt.var(dim=0).mean()
        L_mv_b = self.info_nce(s_img, txt_feat) + s_img.var(dim=0).mean()

        # noise suppression
        n_img_n = F.normalize(n_img, dim=-1)
        n_txt_n = F.normalize(n_txt, dim=-1)
        img_n = F.normalize(img_feat, dim=-1)
        txt_n = F.normalize(txt_feat, dim=-1)

        L_noise = (n_img.pow(2).mean() + n_txt.pow(2).mean()) \
                  + (torch.abs((n_img_n * txt_n).sum(dim=-1)).mean()
                     + torch.abs((n_txt_n * img_n).sum(dim=-1)).mean())

        fused = 0.5 * (s_img + s_txt)
        L_fusion = 0.5 * (L_mv_a + L_mv_b) + L_noise
        return fused, L_fusion


# ============================================================
# ��ģ�ͣ��ؼ��Ķ��������� concat(joint, fused) �����б���
# ============================================================
class MM_GTUNets(nn.Module):
    def __init__(self, args, fold):
        super(MM_GTUNets, self).__init__()
        self.args = args
        self.fold = fold
        self.device = args.device
        self.dropout = args.dropout
        self.edge_drop = args.edge_drop
        self.load_pretrain = True

        # VAE / attention
        self.setup_vae_pretrain()
        self.setup_attenton()

        # ��ģ̬ GTUNet
        self.img_unet = GTUNet(in_channels=args.node_dim, hidden_channels=args.hidden, out_channels=args.out,
                               depth=args.img_depth, edge_dim=1, pool_ratios=args.pool_ratios, dropout=args.dropout)
        self.ph_unet = GTUNet(in_channels=args.node_dim, hidden_channels=args.hidden, out_channels=args.out,
                              depth=args.ph_depth, edge_dim=1, pool_ratios=args.pool_ratios, dropout=args.dropout)

        # 3.1
        self.unc_head_img = GaussianUncertaintyHead(in_dim=args.out, latent_dim=args.out)
        self.unc_head_ph = GaussianUncertaintyHead(in_dim=args.out, latent_dim=args.out)

        # 3.2
        num_tokens = getattr(args, "num_tokens", 8)
        token_dim = getattr(args, "token_dim", max(1, args.out // 2))
        self.tokenizer_img = Tokenizer(in_dim=args.out, num_tokens=num_tokens, token_dim=token_dim)
        self.tokenizer_ph = Tokenizer(in_dim=args.out, num_tokens=num_tokens, token_dim=token_dim)
        self.token_lam = getattr(args, "token_lam", 1e-2)

        # 3.3
        self.two_view_fusion = TwoViewFusion(dim=args.out, temp=getattr(args, "nce_temp", 0.07))

        # loss weights���Ƚ�Ĭ�ϣ�tok ��С��
        self.w_mod = getattr(args, "w_mod", 0.2)
        self.w_tok = getattr(args, "w_tok", 0.01)  # ? Ĭ�ϴ������
        self.w_fus = getattr(args, "w_fus", 0.2)
        self.w_graph = getattr(args, "w_graph", 1.0)

        # ? ����ͷ���� 2*out��concat��
        self.clf = nn.Sequential(
            nn.Linear(args.out * 2, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, args.num_classes)
        )

    # -------- VAE (����) --------
    def setup_vae_pretrain(self):
        self.vae = VAE(self.args)
        self.init_vae_optimizer()
        self.init_vae_save_path()
        self.init_vae_early_stop()

    def setup_attenton(self):
        self.rp_attention = RP_Attention(self.args)
        self.mm_attention = Multimodal_Attention(self.args)

    def init_vae_save_path(self):
        self.vae_save_path = self.args.ckpt_path + "/fold{}_pretrain.pth".format(self.fold)

    def init_vae_optimizer(self):
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=self.args.vae_lr, weight_decay=5e-4)

    def init_vae_early_stop(self):
        self.early_stopping = EarlyStopping(patience=self.args.early_stop, verbose=True)

    def load_vae(self):
        self.vae.load_state_dict(torch.load(self.vae_save_path, map_location=self.device))
        for param in self.vae.parameters():
            param.requires_grad = False

    def train_vae(self, ph_features):
        print("Start pretraining vae...")
        best_loss = 1e50
        best_epo = 0
        ph_features = ph_features.to(self.device)

        for epoch in range(3000):
            self.vae.train()
            self.vae_optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                _, loss = self.vae(ph_features)

            loss.backward()
            self.vae_optimizer.step()

            if epoch % 100 == 0:
                print(f"Epoch: {epoch},\tlr: {self.vae_optimizer.param_groups[0]['lr']:.5f},\tloss: {loss.item():.5f}")

            if best_loss > loss:
                best_loss = loss
                best_epo = epoch
                if self.args.ckpt_path != '':
                    if not os.path.exists(self.args.ckpt_path):
                        os.makedirs(self.args.ckpt_path)
                    torch.save(self.vae.state_dict(), self.vae_save_path)

            self.early_stopping(loss, self.vae)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break

        print(f"\n => Fold {self.fold} best pretrain vae loss {best_loss:.5f}, epoch {best_epo}\n")

    # -------- Graph part (����) --------
    def create_rp_graph(self, affinity_graphs):
        rp_graph, value = self.rp_attention(affinity_graphs)
        return rp_graph, value

    def cal_graph_loss(self, img_embed, ph_embed):
        L = torch.diagflat(torch.sum(self.fused_graph, -1)) - self.fused_graph
        img_smh_loss = self.cal_smh_loss(img_embed, L)
        ph_smh_loss = self.cal_smh_loss(ph_embed, L)

        deg_loss = self.cal_deg_loss()
        img_loss = self.args.smh * img_smh_loss + self.args.deg * deg_loss
        ph_loss = self.args.smh * ph_smh_loss + self.args.deg * deg_loss

        reward_loss = self.cal_reward_loss()
        graph_loss = self.img_weight * img_loss + self.ph_weight * (ph_loss + reward_loss)
        return graph_loss

    def cal_smh_loss(self, embed, L):
        denom = torch.prod(torch.tensor(self.fused_graph.shape, dtype=torch.float, device=embed.device))
        smh_loss = torch.trace(torch.mm(embed.T, torch.mm(L, embed)) / (denom + 1e-8))
        return smh_loss

    def cal_deg_loss(self):
        one = torch.ones(self.fused_graph.size(-1), device=self.device)
        deg_loss = torch.sum(torch.mm(self.fused_graph, one.unsqueeze(-1) + 1e-5).log()) / self.fused_graph.shape[-1]
        return deg_loss

    def cal_reward_loss(self):
        reward_loss = self.args.val * (1 / (self.value + 1e-5))
        return reward_loss

    def forward(self, img_features, ph_features, affinity_graphs):
        img_features = img_features.to(self.device)
        ph_features = ph_features.to(self.device)
        affinity_graphs = affinity_graphs.to(self.device)

        # (A) VAE��������
        if self.load_pretrain:
            self.load_vae()
        ph_features, _vae_loss = self.vae(ph_features)

        # (B) ��ͼ��������
        self.rp_graph, self.value = self.create_rp_graph(affinity_graphs)
        fused_embed = torch.cat((img_features, ph_features), dim=1)
        self.fused_sim = cal_feature_sim(fused_embed)

        self.fused_graph = self.fused_sim * self.rp_graph
        fused_index, fused_attr = dense_to_sparse(self.fused_graph)
        fused_attr = fused_attr.view(-1, 1)

        if self.training and (self.edge_drop > 0):
            fused_index, fused_mask = dropout_edge(fused_index)
            fused_attr = fused_attr[fused_mask]

        # (C) ��ģ̬ͼ������������
        img_embed = self.img_unet(img_features, fused_index, fused_attr)
        ph_embed = self.ph_unet(ph_features, fused_index, fused_attr)

        # (D) ԭ mm_attention��������
        self.joint_embed, self.img_weight, self.ph_weight = self.mm_attention(img_embed, ph_embed)

        # -------- ���� 3.1����ȷ���Զ��� --------
        mu_i, logvar_i = self.unc_head_img(img_embed)
        mu_p, logvar_p = self.unc_head_ph(ph_embed)
        L_mod = symmetric_kl_diag_gaussian(mu_i, logvar_i, mu_p, logvar_p)

        # -------- ���� 3.2��token �������ӣ�Ĭ��������--------
        tok_i = self.tokenizer_img(img_embed)
        tok_p = self.tokenizer_ph(ph_embed)
        L_tok = token_bidir_operator_loss(tok_i, tok_p, lam=self.token_lam)

        # -------- ���� 3.3��multi-view fusion --------
        fused_repr, L_fus = self.two_view_fusion(img_embed, ph_embed)

        # (E) ԭ graph loss��������
        L_graph = self.cal_graph_loss(img_embed, ph_embed)

        # (F) �ܸ�����ʧ������ CE��
        aux_loss = self.w_graph * L_graph + self.w_mod * L_mod + self.w_tok * L_tok + self.w_fus * L_fus

        # ? �������룺concat(joint_embed, fused_repr)
        final_repr = torch.cat([self.joint_embed, fused_repr], dim=-1)
        outputs = self.clf(final_repr)

        loss_dict = {
            "aux": aux_loss.detach(),
            "graph": L_graph.detach(),
            "mod": L_mod.detach(),
            "tok": L_tok.detach(),
            "fus": L_fus.detach(),
        }
        return outputs, aux_loss, loss_dict
