import torchvision


def plot_images(imgs, max_number=256):
    imgs = imgs[:max_number]

    img_grid = torchvision.utils.make_grid(
        imgs, nrow=int(len(imgs) ** 0.5), normalize=True, pad_value=0.5
    )
    img_grid = img_grid.permute(1, 2, 0)

    return img_grid
