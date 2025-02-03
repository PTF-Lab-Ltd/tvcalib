from typing import Optional, Union
import torch
from kornia.geometry.conversions import convert_points_from_homogeneous


class LineCollection:
    def __init__(
        self,
        support: torch.tensor,
        direction_norm: torch.tensor,
        direction: Optional[torch.tensor] = None,
    ):
        """Wrapper class to represent lines by support and direction vectors.

        Args:
            support (torch.tensor): with shape (*, {2,3})
            direction_norm (torch.tensor): with shape (*, {2,3})
            direction (Optional[torch.tensor], optional): Unnormalized direction vector. Defaults to None.
        """
        self.support = support
        self.direction_norm = direction_norm
        self.direction = direction

    def __copy__(self):
        return LineCollection(
            self.support.clone(),
            self.direction_norm.clone(),
            self.direction.clone() if self.direction is not None else None,
        )

    def copy(self):
        return self.__copy__()

    def shape(self):
        return f"support={self.support.shape} direction_norm={self.direction_norm.shape} direction={self.direction.shape if self.direction else None}"

    def __repr__(self) -> str:
        return f"{self.__class__} " + self.shape()


def distance_line_pointcloud_3d(
    e1: torch.Tensor,
    r1: torch.Tensor,
    pc: torch.Tensor,
    reduce: Union[None, str] = None,
) -> torch.Tensor:
    """
    Line to point cloud distance with arbitrary leading dimensions.

    TODO. if cross = (0.0.0) -> distance=0 otherwise NaNs are returned

    https://mathworld.wolfram.com/Point-LineDistance2-Dimensional.html
    Args:
        e1 (torch.Tensor): direction vector of shape (*, B, 1, 3)
        r1 (torch.Tensor): support vector of shape (*, B, 1, 3)
        pc (torch.Tensor): point cloud of shape (*, B, A, 3)
        reduce (Union[None, str]): reduce distance for all points to one using 'mean' or 'min'
    Returns:
        distance of an infinite line to given points, (*, B, ) using reduce='mean' or reduce='min' or (*, B, A) if reduce=False
    """

    num_points = pc.shape[-2]
    _sub = r1 - pc  # (*, B, A, 3)

    cross = torch.cross(e1.repeat_interleave(num_points, dim=-2), _sub, dim=-1)  # (*, B, A, 3)

    e1_norm = torch.linalg.norm(e1, dim=-1)
    cross_norm = torch.linalg.norm(cross, dim=-1)

    d = cross_norm / e1_norm
    if reduce == "mean":
        return d.mean(dim=-1)  # (*, B, )
    elif reduce == "min":
        return d.min(dim=-1)[0]  # (*, B, )

    return d  # (B, A)


def distance_point_pointcloud(points: torch.Tensor, pointcloud: torch.Tensor) -> torch.Tensor:
    """Batched version for point-pointcloud distance calculation"""
    batch_size, T, _, S, N = points.shape
    total_points = batch_size * T * S * N

    # # Debug shapes
    # print(f"Points shape: {points.shape}")
    # print(f"Expected reshape size: {total_points}")

    # Move dimensions before normalization
    points = points.permute(0, 1, 3, 4, 2)  # Now [1,1,4,8,3]

    points[..., :2].div_(points[..., 2:3])
    points_xy = points[..., :2]
    
    # Use broadcasting for distance calculation
    # points_xy: [1,1,4,8,2]
    # pointcloud: [1,1,4,N_star,2]
    diff = points_xy.unsqueeze(-2) - pointcloud.unsqueeze(-3)
    distances = torch.sum(diff * diff, dim=-1)
    
    min_distances = distances.min(dim=-1)[0]  # [1,1,4,8]
    # print(f"Min distances shape: {min_distances.shape}")
    res = min_distances.unsqueeze(2)  # [1,1,1,4]
    # print(res.shape)
    return res
