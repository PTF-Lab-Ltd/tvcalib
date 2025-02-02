from functools import partial
from typing import Tuple
import time
from functools import wraps
import torch
from tqdm.auto import tqdm
from kornia.geometry.conversions import convert_points_to_homogeneous


from tvcalib.cam_modules import CameraParameterWLensDistDictZScore, SNProjectiveCamera
from tvcalib.utils.linalg import distance_line_pointcloud_3d, distance_point_pointcloud


def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__}: {end - start:.4f} seconds")
        return result
    return wrapper


class TVCalibModule(torch.nn.Module):
    def __init__(
        self,
        model3d,
        cam_distr,
        dist_distr,
        image_dim: Tuple[int, int],
        optim_steps: int,
        device="cuda",
        tqdm_kwqargs=None,
        log_per_step=False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.image_height, self.image_width = image_dim
        self.principal_point = (self.image_width / 2, self.image_height / 2)
        self.model3d = model3d
        self.cam_param_dict = CameraParameterWLensDistDictZScore(
            cam_distr, dist_distr, device=device
        )
        phi_hat, psi_hat = self._get_camera_params()
        self.cam = SNProjectiveCamera(
            phi_hat, psi_hat,
            self.principal_point,
            self.image_width, self.image_height,
            device=device,
            nan_check=False,
        )

        self.lens_distortion_active = False if dist_distr is None else True
        self.optim_steps = optim_steps
        self._device = device

        self.optim = torch.optim.AdamW(
            self.cam_param_dict.param_dict.parameters(), lr=0.1, weight_decay=0.01
        )
        self.Scheduler = partial(
            torch.optim.lr_scheduler.OneCycleLR,
            max_lr=0.05,
            total_steps=self.optim_steps,
            pct_start=0.5,
        )

        if self.lens_distortion_active:
            self.optim_lens_distortion = torch.optim.AdamW(
                self.cam_param_dict.param_dict_dist.parameters(), lr=1e-3, weight_decay=0.01
            )
            self.Scheduler_lens_distortion = partial(
                torch.optim.lr_scheduler.OneCycleLR,
                max_lr=1e-3,
                total_steps=self.optim_steps,
                pct_start=0.33,
                optimizer=self.optim_lens_distortion,
            )

        self.tqdm_kwqargs = tqdm_kwqargs
        if tqdm_kwqargs is None:
            self.tqdm_kwqargs = {}

        self.hparams = {"optim": str(
            self.optim), "scheduler": str(self.Scheduler)}
        self.log_per_step = log_per_step

        self.current_step = 0
        self.max_steps = optim_steps

        self.points_buffer = None
        self.reshape_buffer = None

    def reset_scheduler(self):
        self.current_step = 0
        self.scheduler = self.Scheduler(self.optim)
        if self.lens_distortion_active:
            self.scheduler_lens_distortion = self.Scheduler_lens_distortion()


    # @timing_decorator
    def _get_camera_params(self):
        return self.cam_param_dict()

    # @timing_decorator
    def _update_camera(self, phi_hat, psi_hat):
        self.cam.update_params(phi_hat, psi_hat)

        # return SNProjectiveCamera(
        #     phi_hat, psi_hat,
        #     self.principal_point,
        #     self.image_width, self.image_height,
        #     device=self._device,
        #     nan_check=False,
        # )

    # @timing_decorator
    def forward(self, x):

        # individual camera parameters & distortion parameters
        phi_hat, psi_hat = self._get_camera_params() # self.cam_param_dict()

        # cam = SNProjectiveCamera(
        #     phi_hat,
        #     psi_hat,
        #     self.principal_point,
        #     self.image_width,
        #     self.image_height,
        #     device=self._device,
        #     nan_check=False,
        # )

        # Initialize camera
        self._update_camera(phi_hat, psi_hat)


        # (batch_size, num_views_per_cam, 3, num_segments, num_points)
        points_px_lines_true = x["lines__ndc_projected_selection_shuffled"].to(
            self._device)
        batch_size, T_l, _, S_l, N_l = points_px_lines_true.shape

        # project circle points
        points_px_circles_true = x["circles__ndc_projected_selection_shuffled"].to(
            self._device)
        _, T_c, _, S_c, N_c = points_px_circles_true.shape
        assert T_c == T_l

        ####################  line-to-point distance at pixel space ####################
        # start and end point (in world coordinates) for each line segment
        # (3, S_l, 2) to (S_l * 2, 3)
        points3d_lines_keypoints = self.model3d.line_segments
        points3d_lines_keypoints = points3d_lines_keypoints.reshape(
            3, S_l * 2).transpose(0, 1)
        points_px_lines_keypoints = convert_points_to_homogeneous(
            self.cam.project_point2ndc(
                points3d_lines_keypoints, lens_distortion=False)
        )  # (batch_size, t_l, S_l*2, 3)

        if batch_size < self.cam.batch_dim:  # actual batch_size smaller than expected, i.e. last batch
            points_px_lines_keypoints = points_px_lines_keypoints[:batch_size]

        points_px_lines_keypoints = points_px_lines_keypoints.view(
            batch_size, T_l, S_l, 2, 3)

        # -> (batch_size, T_l, 1, S_l, 3)
        lp1 = points_px_lines_keypoints[..., 0, :].unsqueeze(-2)
        # -> (batch_size, T_l, 1, S_l, 3)
        lp2 = points_px_lines_keypoints[..., 1, :].unsqueeze(-2)
        # (batch_size, T, 3, S, N) -> (batch_size, T, 3, S*N) -> (batch_size, T, S*N, 3) -> (batch_size, T, S, N, 3)
        pc = (
            points_px_lines_true.view(batch_size, T_l, 3, S_l * N_l)
            .transpose(2, 3)
            .view(batch_size, T_l, S_l, N_l, 3)
        )

        if self.lens_distortion_active:
            # undistort given points
            pc = pc.view(batch_size, T_l, S_l * N_l, 3)
            pc = pc.detach().clone()
            pc[..., :2] = self.cam.undistort_points(
                pc[..., :2], self.cam.intrinsics_ndc, num_iters=1
            )  # num_iters=1 might be enough for a good approximation
            pc = pc.view(batch_size, T_l, S_l, N_l, 3)

        distances_px_lines_raw = distance_line_pointcloud_3d(
            e1=lp2 - lp1, r1=lp1, pc=pc, reduce=None
        )  # (batch_size, T_l, S_l, N_l)
        distances_px_lines_raw = distances_px_lines_raw.unsqueeze(-3)
        # (..., 1, S_l, N_l,), i.e. (batch_size, T, 1, S_l, N_l)
        ####################  circle-to-point distance at pixel space ####################

        # circle segments are approximated as point clouds of size N_c_star
        points3d_circles_pc = self.model3d.circle_segments
        _, S_c, N_c_star = points3d_circles_pc.shape
        points3d_circles_pc = points3d_circles_pc.reshape(
            3, S_c * N_c_star).transpose(0, 1)
        points_px_circles_pc = self.cam.project_point2ndc(
            points3d_circles_pc, lens_distortion=False)

        if batch_size < self.cam.batch_dim:  # actual batch_size smaller than expected, i.e. last batch
            points_px_circles_pc = points_px_circles_pc[:batch_size]

        if self.lens_distortion_active:
            # (batch_size, T_c, _, S_c, N_c)
            points_px_circles_true = points_px_circles_true.view(
                batch_size, T_c, 3, S_c * N_c
            ).transpose(2, 3)
            points_px_circles_true = points_px_circles_true.detach().clone()
            points_px_circles_true[..., :2] = self.cam.undistort_points(
                points_px_circles_true[...,
                                       :2], self.cam.intrinsics_ndc, num_iters=1
            )
            points_px_circles_true = points_px_circles_true.transpose(2, 3).view(
                batch_size, T_c, 3, S_c, N_c
            )

        distances_px_circles_raw = distance_point_pointcloud(
            points_px_circles_true, points_px_circles_pc.view(
                batch_size, T_c, S_c, N_c_star, 2)
        )

        distances_dict = {
            # (batch_size, T_l, 1, S_l, N_l)
            "loss_ndc_lines": distances_px_lines_raw,
            # (batch_size, T_c, 1, S_c, N_c)
            "loss_ndc_circles": distances_px_circles_raw,
        }
        return distances_dict, self.cam

    def self_optim_batch(self, x, *args, **kwargs):

        # re-initialize lr scheduler for every batch
        if self.current_step >= self.max_steps:
            self.reset_scheduler()

        if not hasattr(self, 'scheduler'):
            self.reset_scheduler()

            # self.scheduler = self.Scheduler(self.optim)
            # if self.lens_distortion_active:
            #     self.scheduler_lens_distortion = self.Scheduler_lens_distortion()


        initial_cam_dict = kwargs.pop('initial_cam_dict', None)
        initial_dist_dict = kwargs.pop('initial_dist_dict', None)
        self.cam_param_dict.initialize(initial_cam_dict, initial_dist_dict)
        self.optim.zero_grad()
        if self.lens_distortion_active:
            self.optim_lens_distortion.zero_grad()

        keypoint_masks = {
            "loss_ndc_lines": x["lines__is_keypoint_mask"].to(self._device),
            "loss_ndc_circles": x["circles__is_keypoint_mask"].to(self._device),
        }
        num_actual_points = {
            "loss_ndc_circles": keypoint_masks["loss_ndc_circles"].sum(dim=(-1, -2)),
            "loss_ndc_lines": keypoint_masks["loss_ndc_lines"].sum(dim=(-1, -2)),
        }
        # print({f"{k} {v}" for k, v in num_actual_points.items()})

        per_sample_loss = {}
        per_sample_loss["mask_lines"] = keypoint_masks["loss_ndc_lines"]
        per_sample_loss["mask_circles"] = keypoint_masks["loss_ndc_circles"]

        per_step_info = {"loss": [], "lr": []}
        # with torch.autograd.detect_anomaly():
        with tqdm(range(self.optim_steps), **self.tqdm_kwqargs) as pbar:
            for step in pbar:

                self.optim.zero_grad()
                if self.lens_distortion_active:
                    self.optim_lens_distortion.zero_grad()

                # forward pass
                distances_dict, cam = self(x)

                # create mask for batch dimension indicating whether distances and loss are computed
                # based on per-sample distance

                # distance calculate with masked input and output
                losses = {}
                for key_dist, distances in distances_dict.items():
                    # for padded points set distance to 0.0
                    # then sum over dimensions (S, N) and divide by number of actual given points
                    distances[~keypoint_masks[key_dist]] = 0.0

                    # log per-point distance
                    per_sample_loss[f"{key_dist}_distances_raw"] = distances

                    # sum px distance over S and number of points, then normalize given the number of annotations
                    distances_reduced = distances.sum(
                        dim=(-1, -2))  # (B, T, 1, S, M) -> (B, T, 1)
                    distances_reduced = distances_reduced / \
                        num_actual_points[key_dist]

                    # num_actual_points == 0 -> set loss for this segment to 0.0 to prevent division by zero
                    distances_reduced[num_actual_points[key_dist] == 0] = 0.0

                    # (B, T, 1) -> (B, T,)
                    distances_reduced = distances_reduced.squeeze(-1)
                    per_sample_loss[key_dist] = distances_reduced

                    # mean over T dimension: (B, T, )-> (B,)
                    loss = distances_reduced.mean(dim=-1)
                    # only relevant for autograd:
                    # sum over batch dimension
                    # --> different batch sizes do not change the per sample loss and its gradients
                    loss = loss.sum()

                    losses[key_dist] = loss

                # each segment and annotation contributes equal to the loss -> no need for weighting segment types
                loss_total_dist = losses["loss_ndc_lines"] + \
                    losses["loss_ndc_circles"]
                loss_total = loss_total_dist

                if self.log_per_step:
                    per_step_info["lr"].append(self.scheduler.get_last_lr())
                    per_step_info["loss"].append(
                        distances_reduced)  # log per sample loss
                if step % 50 == 0:
                    pbar.set_postfix(
                        loss=f"{loss_total_dist.detach().cpu().tolist():.5f}",
                        loss_lines=f'{losses["loss_ndc_lines"].detach().cpu().tolist():.3f}',
                        loss_circles=f'{losses["loss_ndc_circles"].detach().cpu().tolist():.3f}',
                    )

                loss_total.backward(retain_graph=True)
                self.optim.step()

                # Safe scheduler step
                if self.current_step < self.max_steps:
                    self.scheduler.step()
                    self.current_step += 1

                if self.lens_distortion_active:
                    self.optim_lens_distortion.step()
                    self.scheduler_lens_distortion.step()

        per_sample_loss["loss_ndc_total"] = torch.sum(
            torch.stack([per_sample_loss[key_dist]
                        for key_dist in distances_dict.keys()], dim=0),
            dim=0,
        )

        if self.log_per_step:
            per_step_info["loss"] = torch.stack(
                per_step_info["loss"], dim=-1
            )  # (n_steps, batch_dim, temporal_dim)
            per_step_info["lr"] = torch.tensor(per_step_info["lr"])
        return per_sample_loss, cam, per_step_info
