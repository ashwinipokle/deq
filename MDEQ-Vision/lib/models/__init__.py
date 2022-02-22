from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import models.mdeq
import models.mdeq_core

#Diffusion models
import models.mdeq_xt
import models.mdeq_core_xt
import models.mdeq_swish
import models.mdeq_core_swish
import models.unet
import models.mdeq_swish_alt
import models.mdeq_core_swish_alt
import models.mdeq_core_swish_attn
import models.mdeq_swish_attn
import models.mdeq_core_relu_alt
import models.mdeq_relu_alt

# Some HRNet models
import models.hrnet
import models.hrnet_swish
import models.hrnet_res
import models.hrnet_swish_gn

# MDEQ modified to be more like UNet
# These two models have connections across all the branches
import models.mdeq_core_swish_attn_unet
import models.mdeq_swish_attn_unet

# These models that only connect to higher and lower resolutions