import paddle


def _repeat_(tensor, n_samples):
    return paddle.concat([tensor for _ in range(n_samples)], axis=0).reshape(
        (n_samples, *tensor.shape)
    )


def append_2d_grid_positional_encoding(
    input_tensor, grid_boundaries=[[0, 1], [0, 1]], channel_dim=1
):
    """
    Appends grid positional encoding to an input tensor, concatenating as additional dimensions along the channels
    """
    shape = list(input_tensor.shape)
    shape.pop(channel_dim)
    n_samples, height, width = shape
    # xt = torch.linspace(grid_boundaries[0][0], grid_boundaries[0][1], height + 1)[:-1]
    xt = paddle.linspace(grid_boundaries[0][0], grid_boundaries[0][1], height + 1)[:-1]
    # yt = torch.linspace(grid_boundaries[1][0], grid_boundaries[1][1], width + 1)[:-1]
    yt = paddle.linspace(grid_boundaries[1][0], grid_boundaries[1][1], width + 1)[:-1]

    # grid_x, grid_y = torch.meshgrid(xt, yt, indexing='ij')
    grid_x, grid_y = paddle.meshgrid(xt, yt, indexing="ij")

    input_tensor = paddle.concat(
        (
            input_tensor,
            # grid_x.repeat(n_samples, 1, 1).unsqueeze(channel_dim),
            _repeat_(grid_x, n_samples).unsqueeze(channel_dim),
            # grid_y.repeat(n_samples, 1, 1).unsqueeze(channel_dim)),
            _repeat_(grid_y, n_samples).unsqueeze(channel_dim),
        ),
        dim=1,
    )

    return input_tensor


def get_grid_positional_encoding(
    input_tensor, grid_boundaries=[[0, 1], [0, 1]], channel_dim=1
):
    """
    Appends grid positional encoding to an input tensor, concatenating as additional dimensions along the channels
        """
    shape = list(input_tensor.shape)
    if len(shape) == 2:
        height, width = shape
    else:
        _, height, width = shape

    xt = paddle.linspace(grid_boundaries[0][0], grid_boundaries[0][1], height + 1)[:-1]
    yt = paddle.linspace(grid_boundaries[1][0], grid_boundaries[1][1], width + 1)[:-1]

    grid_x, grid_y = paddle.meshgrid(xt, yt, indexing="ij")

    if len(shape) == 2:
        # grid_x = grid_x.repeat(1, 1).unsqueeze(channel_dim)
        # paddle doesn't have repeat api, use tile instead
        grid_x = grid_x.unsqueeze(channel_dim)
        grid_y = grid_y.unsqueeze(channel_dim)
    else:
        grid_x = grid_x.unsqueeze(0).unsqueeze(channel_dim)
        grid_y = grid_y.unsqueeze(0).unsqueeze(channel_dim)

    return grid_x, grid_y
