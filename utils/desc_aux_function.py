import torch

def _meshgrid(height, width):
   
    x_t = torch.matmul(torch.ones([height, 1]), torch.linspace(-1.0,1.0,width).unsqueeze(1).transpose(1,0))
    y_t = torch.matmul(torch.linspace(-1.0,1.0,height).unsqueeze(1), torch.ones([1, width]) ) 
    
    x_t_flat = torch.reshape(x_t, (1, -1))
    y_t_flat = torch.reshape(y_t, (1, -1))
   
    ones = torch.ones_like(x_t_flat)
    grid = torch.cat([x_t_flat, y_t_flat, ones], dim=0)
    return grid


def transformer_crop(images, out_size, batch_inds, kpts_xy, kpts_scale=None, kpts_ori=None, thetas=None,
                     name='SpatialTransformCropper'):
    # images : [B,C,H,W]
    # out_size : (out_width, out_height)
    # batch_inds : [B*K,] torch.int32 [0,B)
    # kpts_xy : [B*K,2] torch.float32 or whatever
    # kpts_scale : [B*K,] torch.float32
    # kpts_ori : [B*K,2] torch.float32 (cos,sin)
    if isinstance(out_size, int):
        out_width = out_height = out_size
    else:
        out_width, out_height = out_size
    hoW = out_width // 2
    hoH = out_height // 2

    num_batch, C, height, width = images.shape
    num_kp = kpts_xy.shape[0]

    zero = torch.zeros([], dtype=torch.int32)
    max_y = torch.tensor(height - 1).to(torch.int32)
    max_x = torch.tensor(width - 1).to(torch.int32)

    grid = _meshgrid(out_height, out_width).to(kpts_xy.device)  # normalized -1~1
    grid = grid.unsqueeze(0)
    grid = torch.reshape(grid, [-1])
    grid = torch.tile(grid, [num_kp])
    grid = torch.reshape(grid, [num_kp, 3, -1])

    # create 6D affine from scale and orientation
    # [s, 0, 0]   [cos, -sin, 0]
    # [0, s, 0] * [sin,  cos, 0]
    # [0, 0, 1]   [0,    0,   1]

    if thetas is None:
        thetas = torch.eye(2, 3, dtype=torch.float32).to(kpts_xy.device)
        thetas = torch.tile(thetas, [num_kp, 1, 1])
        if kpts_scale is not None:
            thetas = thetas * kpts_scale[:, None, None]
        ones = torch.tile(torch.tensor([[[0, 0, 1]]], dtype=torch.float32), [num_kp, 1, 1]).to(kpts_xy.device)
        thetas = torch.cat([thetas, ones], dim=1)  # [num_kp, 3,3]
        
        if kpts_ori is not None:
            cos = kpts_ori[:, 0]  # [num_kp, 1]
            sin = kpts_ori[:, 1]
            zeros = torch.zeros_like(cos).to(kpts_xy.device)
            ones = torch.ones_like(cos).to(kpts_xy.device)
            R = torch.cat([cos, -sin, zeros, sin, cos, zeros, zeros, zeros, ones], dim=-1)
            R = torch.reshape(R, [-1, 3, 3])
            thetas = torch.matmul(thetas, R)
    # Apply transformation to regular grid
    T_g = torch.matmul(thetas, grid)  # [num_kp,3,3] * [num_kp,3,H*W]
    x = T_g[:, 0, :]  # [num_kp,1,H*W]
    y = T_g[:, 1, :]

    # unnormalization [-1,1] --> [-out_size/2,out_size/2]
    x = x * out_width / 2.0
    y = y * out_height / 2.0

    kp_x_ofst = kpts_xy[:,0].unsqueeze(1)# [B*K,1,1]
    kp_y_ofst = kpts_xy[:,1].unsqueeze(1)# [B*K,1,1]

    # centerize on keypoints
    x = x + kp_x_ofst
    y = y + kp_y_ofst
    x = torch.reshape(x, [-1])  # num_kp*out_height*out_width
    y = torch.reshape(y, [-1])

    # interpolation
    x0 = torch.floor(x).to(torch.int32)
    x1 = x0 + 1
    y0 = torch.floor(y).to(torch.int32)
    y1 = y0 + 1

    x0 = torch.clamp(x0, zero, max_x)
    x1 = torch.clamp(x1, zero, max_x)
    y0 = torch.clamp(y0, zero, max_y)
    y1 = torch.clamp(y1, zero, max_y)

    dim2 = width
    dim1 = width * height
    base = torch.tile(batch_inds[:, None], [1, out_height * out_width])  # [B*K,out_height*out_width]
    base = torch.reshape(base, [-1]) * dim1
    base_y0 = base + y0 * dim2
    base_y1 = base + y1 * dim2
    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1

    im_flat = torch.reshape(images.permute(0,2,3,1), [-1, C])  # [B*height*width,C]
    im_flat = im_flat.to(torch.float32)
    
    Ia = torch.gather(im_flat, 0, idx_a.to(torch.int64).unsqueeze(1).repeat(1,C))
    Ib = torch.gather(im_flat, 0, idx_b.to(torch.int64).unsqueeze(1).repeat(1,C))
    Ic = torch.gather(im_flat, 0, idx_c.to(torch.int64).unsqueeze(1).repeat(1,C))
    Id = torch.gather(im_flat, 0, idx_d.to(torch.int64).unsqueeze(1).repeat(1,C))

    x0_f = x0.to(torch.float32)
    x1_f = x1.to(torch.float32)
    y0_f = y0.to(torch.float32)
    y1_f = y1.to(torch.float32)
    
    wa = ((x1_f - x) * (y1_f - y)).unsqueeze(1)
    wb = ((x1_f - x) * (y - y0_f)).unsqueeze(1)
    wc = ((x - x0_f) * (y1_f - y)).unsqueeze(1)
    wd = ((x - x0_f) * (y - y0_f)).unsqueeze(1)

    output = wa * Ia + wb * Ib + wc * Ic + wd * Id
    output = torch.reshape(output, [num_kp, out_height, out_width, C])
    output = output.permute(0,3,1,2)
    #output.set_shape([batch_inds.shape[0], out_height, out_width, images.shape[-1]])
    return output


def build_patch_extraction(kpts, batch_inds, images, kpts_scale, name='PatchExtract', patch_size=32):
    patches = transformer_crop(images, patch_size, batch_inds, kpts, kpts_scale=kpts_scale)

    return patches
