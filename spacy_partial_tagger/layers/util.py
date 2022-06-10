import torch


def get_mask(
    lengths: torch.Tensor, max_length: int, device: torch.device
) -> torch.Tensor:
    """Converts length tensor to mask tensor."""
    return (
        torch.arange(
            max_length,
            device=device,
        )[None, :]
        < lengths[:, None]
    )
