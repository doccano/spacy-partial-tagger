import torch


def get_mask(
    lengths: torch.Tensor, max_length: int, device: torch.device
) -> torch.Tensor:
    return (
        torch.arange(
            max_length,
            device=device,
        )[None, :]
        < lengths[:, None]
    )
