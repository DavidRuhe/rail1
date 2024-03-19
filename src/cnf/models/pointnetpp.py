import torch
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils
from torch import nn
import math


def build_shared_mlp(mlp_spec, bn: bool = True):
    layers = []
    for i in range(1, len(mlp_spec)):
        layers.append(
            nn.Conv2d(mlp_spec[i - 1], mlp_spec[i], kernel_size=1, bias=not bn)
        )
        if bn:
            layers.append(nn.BatchNorm2d(mlp_spec[i]))
        layers.append(nn.ReLU(True))

    return nn.Sequential(*layers)


# def custom_ips(xyz, n):

#     ix = torch.zeros((len(xyz)), dtype=torch.int64, device=xyz.device)
#     indices = [ix]
#     B, N = xyz.shape[:2]
#     range_tensor = torch.arange(N, device=xyz.device).unsqueeze(0).repeat(B, 1)
#     mask = range_tensor != ix.unsqueeze(1)

#     for i in range(n - 1):
#         points = torch.gather(xyz, 1, ix.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 3))
#         d = torch.cdist(points, xyz).squeeze(1)

#         d *= mask

#         ix = torch.argmax(d, dim=1)

#         indices.append(ix)
#     mask = mask & (range_tensor != ix.unsqueeze(1))

#     return torch.stack(indices, dim=1).to(torch.int32)

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N).to(device) * 1e10
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    farthest = torch.zeros(B, dtype=torch.long, device=device)

    centroid_locs = torch.zeros(B, npoint, 3, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        centroid_locs[:, i: i + 1] = centroid
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance  # Smaller such that we select a new point
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids.to(torch.int32), centroid_locs


class PointnetSAModule(nn.Module):
    def __init__(self, npoint, radii, nsamples, mlps, bn=True, use_xyz=True):
        super(PointnetSAModule, self).__init__()
        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()

        # Handle the case where the inputs are not lists (single scale)
        if not isinstance(radii, list):
            radii = [radii]
        if not isinstance(nsamples, list):
            nsamples = [nsamples]
        if not isinstance(mlps[0], list):
            mlps = [mlps]

        assert (
            len(radii) == len(nsamples) == len(mlps)
        ), "radii, nsamples, and mlps lists must be of equal length"

        for i in range(len(radii)):
            radius, nsample, mlp_spec = radii[i], nsamples[i], mlps[i]
            if use_xyz:
                mlp_spec[0] += 3
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                if npoint
                else pointnet2_utils.GroupAll(use_xyz)
            )
            self.mlps.append(build_shared_mlp(mlp_spec, bn))

    def forward(self, xyz, features):
        new_features_list = []
        xyz_flipped = xyz.transpose(1, 2).contiguous()
        # new_xyz = (
        #     pointnet2_utils.gather_operation(
        #         xyz_flipped, pointnet2_utils.furthest_point_sample(xyz, self.npoint)
        #     )
        #     .transpose(1, 2)
        #     .contiguous()
        #     if self.npoint
        #     else None
        # )
        with torch.no_grad():
                xyz = torch.cat([torch.mean(xyz, dim=1, keepdim=True), xyz], dim=1)
                if self.npoint is not None:
                    centroids, centroid_locs = farthest_point_sample(xyz, self.npoint)
                # new_xyz = (
                #     pointnet2_utils.gather_operation(
                #         xyz_flipped, centroids
                #     )
                #     .transpose(1, 2)
                #     .contiguous()
                #     if self.npoint
                #     else None
                # )
                    new_xyz = centroid_locs
                else:
                    new_xyz = None
                xyz = xyz[:, 1:].contiguous()


        # Idea: look which K points are closest to the new centroids.
        # Process them, and agggregate them to the new centroids.
        for grouper, mlp in zip(self.groupers, self.mlps):
            new_features = grouper(xyz, new_xyz, features)  # (B, C, npoint, nsample)
            new_features = mlp(new_features)  # (B, mlp[-1], npoint, nsample)
            # new_features = F.(
            #     new_features, kernel_size=[1, new_features.size(3)]
            # )  # (B, mlp[-1], npoint, 1)
            # new_features = 1 / math.sqrt(new_features.size(3)) * torch.sum(
            #     new_features, dim=3, keepdim=False
            # )  # (B, mlp[-1], npoint, 1)
            new_features = torch.max(new_features, 3)[0]  # (B, mlp[-1], npoint, 1
            # new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1)


lr_clip = 1e-5
bnm_clip = 1e-2


class PointNetPPClassification(nn.Module):
    def __init__(self, use_xyz=True):
        super().__init__()

        self.use_xyz = use_xyz

        self._build_model()

    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=512,
                radii=[0.2],
                nsamples=[64],
                mlps=[[0, 64, 64, 128]],
                use_xyz=self.use_xyz,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radii=[0.4],
                nsamples=[64],
                mlps=[[128, 128, 128, 256]],
                use_xyz=self.use_xyz,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                mlps=[[256, 256, 512, 1024]],
                use_xyz=self.use_xyz,
                npoint=None,
                radii=None,
                nsamples=None,
            )
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, 40),
        )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud):
        r"""
        Forward pass of the network

        Parameters
        ----------
        pointcloud: Variable(torch.cuda.FloatTensor)
            (B, N, 3 + input_channels) tensor
            Point cloud to run predicts on
            Each point in the point-cloud MUST
            be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        for module in self.SA_modules:
            xyz, features = module(xyz, features)

        return self.fc_layer(features.squeeze(-1))

    def training_step(self, batch, batch_idx):
        pc, labels = batch

        logits = self.forward(pc)
        loss = F.cross_entropy(logits, labels)
        with torch.no_grad():
            acc = (torch.argmax(logits, dim=1) == labels).float().mean()

        log = dict(train_loss=loss, train_acc=acc)

        return dict(loss=loss, log=log, progress_bar=dict(train_acc=acc))

    def validation_step(self, batch, batch_idx):
        pc, labels = batch

        logits = self.forward(pc)
        loss = F.cross_entropy(logits, labels)
        acc = (torch.argmax(logits, dim=1) == labels).float().mean()

        return dict(val_loss=loss, val_acc=acc)

    def validation_end(self, outputs):
        reduced_outputs = {}
        for k in outputs[0]:
            for o in outputs:
                reduced_outputs[k] = reduced_outputs.get(k, []) + [o[k]]

        for k in reduced_outputs:
            reduced_outputs[k] = torch.stack(reduced_outputs[k]).mean()

        reduced_outputs.update(
            dict(log=reduced_outputs.copy(), progress_bar=reduced_outputs.copy())
        )

        return reduced_outputs

    # def configure_optimizers(self):
    #     lr_lbmd = lambda _: max(
    #         self.hparams["optimizer.lr_decay"]
    #         ** (
    #             int(
    #                 self.global_step
    #                 * self.hparams["batch_size"]
    #                 / self.hparams["optimizer.decay_step"]
    #             )
    #         ),
    #         lr_clip / self.hparams["optimizer.lr"],
    #     )
    #     bn_lbmd = lambda _: max(
    #         self.hparams["optimizer.bn_momentum"]
    #         * self.hparams["optimizer.bnm_decay"]
    #         ** (
    #             int(
    #                 self.global_step
    #                 * self.hparams["batch_size"]
    #                 / self.hparams["optimizer.decay_step"]
    #             )
    #         ),
    #         bnm_clip,
    #     )

    #     optimizer = torch.optim.Adam(
    #         self.parameters(),
    #         lr=self.hparams["optimizer.lr"],
    #         weight_decay=self.hparams["optimizer.weight_decay"],
    #     )
    #     lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lambda=lr_lbmd)
    #     bnm_scheduler = BNMomentumScheduler(self, bn_lambda=bn_lbmd)

    #     return [optimizer], [lr_scheduler, bnm_scheduler]

    # def prepare_data(self):
    #     train_transforms = transforms.Compose(
    #         [
    #             d_utils.PointcloudToTensor(),
    #             d_utils.PointcloudScale(),
    #             d_utils.PointcloudRotate(),
    #             d_utils.PointcloudRotatePerturbation(),
    #             d_utils.PointcloudTranslate(),
    #             d_utils.PointcloudJitter(),
    #             d_utils.PointcloudRandomInputDropout(),
    #         ]
    #     )

    #     self.train_dset = ModelNet40Cls(
    #         self.hparams["num_points"], transforms=train_transforms, train=True
    #     )
    #     self.val_dset = ModelNet40Cls(
    #         self.hparams["num_points"], transforms=None, train=False
    #     )

    # def _build_dataloader(self, dset, mode):
    #     return DataLoader(
    #         dset,
    #         batch_size=self.hparams["batch_size"],
    #         shuffle=mode == "train",
    #         num_workers=4,
    #         pin_memory=True,
    #         drop_last=mode == "train",
    #     )

    # def train_dataloader(self):
    #     return self._build_dataloader(self.train_dset, mode="train")

    # def val_dataloader(self):
    #     return self._build_dataloader(self.val_dset, mode="val")
