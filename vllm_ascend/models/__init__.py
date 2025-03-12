from vllm import ModelRegistry


def register_model():
    from .deepseek_v2 import CustomDeepseekV2ForCausalLM, CustomDeepseekV3ForCausalLM  # noqa: F401

    ModelRegistry.register_model(
        "DeepseekV2ForCausalLM",
        "vllm_ascend.models.deepseek_v2:CustomDeepseekV2ForCausalLM")

    ModelRegistry.register_model(
        "DeepseekV3ForCausalLM",
        "vllm_ascend.models.deepseek_v2:CustomDeepseekV3ForCausalLM")
