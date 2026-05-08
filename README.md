# L1 Residual-Aware Selective Pruning for 3D Gaussian Splatting 

## Introduction 
  
This project is used to reproduce. The paper proposes a general and theoretically grounded method for assessing the importance of scene representation primitives. This project mainly targets 3DGS.

### Importance Score of Scene Representation Primitives $i$ in a single view $\theta$

 General Form:

 $score(\theta ,i)=\sum_{p\in P}(\frac{\alpha_i^\theta(p)}{1-\alpha_i^\theta(p)})^2 $ 
 
 $\alpha_i^\theta(p)$ is the $\alpha$ value at pixel $p$ under view $\theta$, $\alpha_i$ can be any scene representation primitives(point, Gaussian, triangle......)
 
 Approximate Form:

 $score(\theta ,i)=\iint_\Omega  (\frac{\alpha_i^\theta(x)}{1-\alpha_i(x)^\theta})^2 dx$
 
 In 3DGS:
$\alpha=oG^{2D}$

### Prune Condition
$E_{\theta\sim U(view_i)}(score(\theta,i))<\tau$

 where $U(.)$  is a uniform distribution. $view_i$ is the set of views in the training set that contain primitive(Gaussians, triangle.......) $i$.

## Code Structure

We only made **4 minor modifications** to the official 3DGS version.

| # | File | Modification | Purpose |
|---|------|------------|---------|
| 1 | `backward.cu` | Added `atomicAdd(&(dL_dmean2D[global_id].z), fabs(1.f/2.f*alpha*alpha/(1-alpha)/(1-alpha)))` | Compute the score in the **general form** |
| 2 | `gaussian_model.py` | Added `compute_score_and_mask` function | Compute the importance score of each Gaussian from a single view (**approximate form**) and its mask (determining whether it is within the view frustum) |
| 3 | `gaussian_model.py` | Added `prune_points_only` function | Prune Gaussians |
| 4 | `train.py` | Overall implementation | Integrate the complete pipeline |

---

**Note:** The score computed in modification #2 is specifically used for the **approximate form** However, we still utilize it to compute the mask for determining whether a Gaussian is within the view frustum..
##  Installation

Installation steps are identical to the official 3DGS, but diff-gaussian-rasterization requires utilizing the module we provide.

## Running

###  Approximate Form (used in the paper)
```shell
python train.py -s <path to COLMAP or NeRF Synthetic dataset> --eval --approximate
```
###  General Form (this method is widely applicable and not limited to 3DGS).
Under the same parameters, this form results in a slightly higher number of Gaussians being pruned.
```shell
python train.py -s <path to COLMAP or NeRF Synthetic dataset> --eval 
```


<span style="font-weight: bold;">Extra Arguments for train.py</span>

  #### --tau_approximate
  tau value when using the approximate form, default: 0.1
  #### --tau_general
  tau value when using the general form, default: 0.1
 




##   Applied to other special splatting methods
**Note:** The methods below all use the general form; theoretically, the approximate form can also be used, provided it can be derived or easily implemented.

### Mip-Splatting


In `gaussian_model.py`, modify the function `add_densification_stats` as follows:

**From:**
```python
self.xyz_gradient_accum_abs[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,2:], dim=-1, keepdim=True)
self.xyz_gradient_accum_abs_max[update_filter] = torch.max(self.xyz_gradient_accum_abs_max[update_filter], torch.norm(viewspace_point_tensor.grad[update_filter,2:], dim=-1, keepdim=True))
```
**To:**
```Python
self.xyz_gradient_accum_abs[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,2], dim=-1, keepdim=True)
self.xyz_gradient_accum_abs_max[update_filter] = torch.max(self.xyz_gradient_accum_abs_max[update_filter], torch.norm(viewspace_point_tensor.grad[update_filter,2], dim=-1, keepdim=True))
```
**Note:** This prevents the score from being used for densification.
  
The other steps can largely be followed directly; note that the pruning function and main pipeline may require small modification.
  
  ```shell
  python train.py -s <path to COLMAP or NeRF Synthetic dataset> --eval -r -1 --kernel_size = 0.1 
  ```
 Parameter $\tau=0.05$

### AAA-Gaussians

  The overall idea is similar, but slight modifications may be required.
  
  **Note:** Densification is performed entirely for the first 15K iterations. We disable noise perturbation after 15K iterations during pruning (retaining it seems to have no effect either).
  ```shell
  python train.py --splatting_config configs/aaa.json -s <path to COLMAP or NeRF Synthetic dataset> --eval --cap_max <max number per scene> 
  ```
  --cap_max
  
  ```
  bicycle  5000000
  bonsai   2000000
  counter  2000000
  flowers  3500000
  ```
  Parameter $\tau=0.02$
### Triangle-Splatting
  The overall idea is consistent; However, the center position of the Triangle needs to be computed. Our calculation method is:
   
  Add `xyz_mean = self._triangles_points.mean(dim=1)` after `focal_y = height / (2 * tanfovy)`  in function compute_score_and_mask.

  and change `xyz_world_homo = torch.cat([self._xyz, torch.ones_like(self._xyz[:, :1])], dim=-1).cuda()`   
  
  to `xyz_world_homo = torch.cat([xyz_mean, torch.ones_like(xyz_mean[:, :1])], dim=-1).cuda()`.

  Densification is performed entirely for the first 15K iterations. You can apply their own pruning function.
  ```shell
  python train.py -s <path to COLMAP or NeRF Synthetic dataset>  --eval
  ```
  For outdoor scenes, add `--outdoor` as well.

  Parameter $\tau=0.02$


