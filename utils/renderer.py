import torch
from torch import nn
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRasterizer,
)


#这个工作根本就不是blend，这个就是直接选择了每个像素最近的那个面片对应的颜色
#然后将图像中的背景设置成常量，再将三通道的图像扩展一个维度alpha通道，如果是背景像素那么alpha通道值等于0，非背景像素alpha通道值不等于0
def hard_channel_blend(
    colors: torch.Tensor, fragments,
) -> torch.Tensor:
    """
    Naive blending of top K faces to return an C+1 image
    Args:
        colors: (N, H, W, K, C) RGB color for each of the top K faces per pixel.
        fragments: the outputs of rasterization. From this we use
            - pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
              of the faces (in the packed representation) which
              overlap each pixel in the image. This is used to
              determine the output shape.
    Returns:
        RGBA pixel_channels: (N, H, W, C+1)
    """
    N, H, W, K = fragments.pix_to_face.shape#N表示批量的大小， H和W表示图像的大小，K表示一个像素和K个面片相关， C表示相机的通道数
    device = fragments.pix_to_face.device

    # Mask for the background.
    #如果第一个面片的索引小于 0（< 0），则说明该像素位置没有被任何面片覆盖，属于背景。因此，is_background 张量中相应位置的值为 True。
    is_background = fragments.pix_to_face[..., 0] < 0  # (N, H, W)

    #background_color = 是一个三维向量
    background_color = torch.ones(colors.shape[-1], dtype=colors.dtype, device=colors.device)

    # Find out how much background_color needs to be expanded to be used for masked_scatter.
    #图像上有多少个像素点是背景
    num_background_pixels = is_background.sum()

    # Set background color.将背景颜色设置为指定颜色！！！！！！！
    #这段代码的目的是通过 masked_scatter 将背景像素的颜色设置为指定的背景颜色 background_color。这样可以确保背景区域的颜色统一，
    #而前景区域的颜色根据原始的 colors 张量保持不变。
    #选择最近那个面片对应的颜色，然后将背景赋值成指定的颜色
    pixel_colors = colors[..., 0, :].masked_scatter(
        is_background[..., None],#将is_background形状从 (N, H, W) 扩展为 (N, H, W, 1)，
        background_color[None, :].expand(num_background_pixels, -1),
    )  # (N, H, W, C)

    # Concat with the alpha channel.
    #background像素 alpha设置为0
    #非background像素，alpha设置为1
    alpha = (~is_background).type_as(pixel_colors)[..., None]

    return torch.cat([pixel_colors, alpha], dim=-1)  # (N, H, W, C+1)

#
class SimpleShader(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        texels = meshes.sample_textures(fragments)#pytorch3d的接口函数
        images = hard_channel_blend(texels, fragments)#这个是用户自己定义的结果
        return images


class MeshRendererWithDepth(nn.Module):
    def __init__(self, rasterizer, shader):
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)
        return images, fragments.zbuf


class Renderer(nn.Module):
    def __init__(self):
        super().__init__()
        self.raster_settings = None

    def set_rasterization(self, cameras):
        image_size = tuple(cameras.image_size[0].detach().cpu().numpy())
        self.raster_settings = RasterizationSettings(
            image_size=(int(image_size[0]), int(image_size[1])),
            blur_radius=0.0,
            faces_per_pixel=1,
        )
    #外部调用的主入口函数
    def forward(self, input):
        mesh = input["mesh"]
        cameras = input["cameras"]
        if self.raster_settings is None:
            self.set_rasterization(cameras)
        #给render设置光栅化和shader方法
        #光栅化 = 直接调用的pytorch3d的方法
        #shader = 用户自己定义的类
        mesh_renderer = MeshRendererWithDepth(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=self.raster_settings
            ),
            shader=SimpleShader()
        )
        images, depth = mesh_renderer(mesh)#在调用了MeshRendererWithDepth的forward方法
        return images, depth


#这个函数没有被使用
class RendererBev(nn.Module):
    def __init__(self):
        super().__init__()
        self.raster_settings = None
        self.image_size = tuple((640, 1024))  # FOV cameras do not have image_size

    def set_rasterization(self):
        image_size = self.image_size
        self.raster_settings = RasterizationSettings(
            image_size=(int(image_size[0]), int(image_size[1])),
            blur_radius=0.0,
            faces_per_pixel=1,
        )

    def forward(self, input):
        mesh = input["mesh"]
        cameras = input["cameras"]
        if self.raster_settings is None:
            self.set_rasterization()

        mesh_renderer = MeshRendererWithDepth(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=self.raster_settings
            ),
            shader=SimpleShader()
        )
        images, depth = mesh_renderer(mesh)
        return images, depth
