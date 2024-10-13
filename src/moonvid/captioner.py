import os
from .utils import load_models, extract_frames, latency
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import Union

def caption_frames(
    image_frames: list,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
) -> list:

    captions = []

    for frame in image_frames:
        enc_image = model.encode_image(
            frame
        )  # encode image with vision encoder(moondream uses SigLip)
        frame_caption = model.answer_question(
            enc_image, "briefly describe this image", tokenizer
        )  # generate caption

        captions.append(frame_caption)

    print(f'moondream captions -> {len(captions)}')
    return captions


# captions = caption_frames(image_frames)


def refine_captions(captions: list, llm_pipeline: pipeline) -> str:
    single_cap = ". ".join(captions)
    messages = [
        {
            "role": "system",
            "content": "You are a summary chatbot who summarizes and arranges several image captions, which are of a video sequence, into one flowing caption",
        },
        {"role": "user", "content": f"{single_cap}"},
    ]

    outputs = llm_pipeline(
        messages,
        max_new_tokens=256,
    )

    llm_caption = outputs[0]["generated_text"][-1]

    return llm_caption


@latency
def moonvid_captioner(video_path: Union[str, os.PathLike]): # main pipeline unifying all functions
    md_model, md_tokenizer, llama_pipe = load_models() # load model weights
    image_frames = extract_frames(video_path) # extract video frames and load PIL format
    moondream_captions = caption_frames(image_frames, md_model, md_tokenizer) # use moondream to caption each frame
    
    llm_refined_caption = refine_captions(moondream_captions, llm_pipeline=llama_pipe) # use llama-1b to refine the caption into one long sentence
    
    return llm_refined_caption