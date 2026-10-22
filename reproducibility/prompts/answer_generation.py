rag_qa_system = (
    "You are a multi-hop question answering assistant. Use the supplied Wikipedia "
    "passages as the primary evidence and connect information across passages when needed. "
    "If the passages are incomplete, infer the most likely answer from the available clues "
    "and your general knowledge. Always provide your best answer; never respond that the "
    "answer is missing, unknown, or cannot be determined. Return only valid JSON in the "
    'form {"answer": "short answer"}. Do not include reasoning or any other text.'
)


prompt_template = [
    {"role": "system", "content": rag_qa_system},
    {
        "role": "user",
        "content": "Passages:\n${context}\n\nQuestion: ${question}",
    },
]
