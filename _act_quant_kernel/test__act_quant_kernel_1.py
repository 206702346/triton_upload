from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from _act_quant_kernel import act_quant

if __name__ == "__main__":

    device = torch.device("npu" if torch.npu.is_available() else "cpu")

    x = torch.randn(2, 4, 256, device=device, dtype=torch.float32)
    x = x.contiguous()

    block_size = 128

    y, s = act_quant(x, block_size=block_size)

    def ref_act_quant(x, block_size):

        orig_shape = x.shape
        x_reshaped = x.view(-1, block_size)

        amax = torch.max(torch.abs(x_reshaped), dim=1, keepdim=True).values
        amax = torch.clamp(amax, min=1e-4)
        scale = amax / 448.0

        y = x_reshaped / scale
        y = torch.clamp(y, min=-448.0, max=448.0)

        y = y.view(orig_shape)
        scale = scale.view(*orig_shape[:-1], -1)
        
        return y.half(), scale.squeeze(-1)

    y_ref, s_ref = ref_act_quant(x, block_size)

    torch.testing.assert_close(y, y_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(s, s_ref, rtol=1e-2, atol=1e-2)
    
    print("All tests passed!")

    y2, s2 = act_quant(x, block_size=block_size, scale_fmt="round")

    assert y2.shape == x.shape
    assert s2.shape == (*x.shape[:-1], x.shape[-1] // block_size)
    
    print("Scale format test passed!")