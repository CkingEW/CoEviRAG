"""CoEviRAG uses only the published GPT-OSS prompt family."""


def qa_prompt_for_model(model_name, dataset, template_manager):
    if "gpt-oss" not in str(model_name).lower():
        raise ValueError("The released reader protocol is GPT-OSS-120B")
    return "rag_qa_gptoss"
