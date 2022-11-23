from ...utils import is_torch_available, is_transformers_available


if is_transformers_available() and is_torch_available():
    from .modeling_gpt2_optimus import GPT2OptimusForLatentConnector
    from .modeling_text_unet import UNetFlatConditionModel
    from .pipeline_versatile_diffusion import VersatileDiffusionPipeline
    from .pipeline_versatile_diffusion_dual_guided import VersatileDiffusionDualGuidedPipeline
    from .pipeline_versatile_diffusion_image_to_text import VersatileDiffusionImageToTextPipeline
    from .pipeline_versatile_diffusion_image_variation import VersatileDiffusionImageVariationPipeline
