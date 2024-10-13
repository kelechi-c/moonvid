import cv2, os, torch, time 
from PIL import Image as pillow
from typing import Union
from functools import wraps
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

local_model_path = "moondream2"
local_tokenizer_path = "moondream2_tokenizer"
llama_model_path = "llama_3-2_1B"
model_id = "vikhyatk/moondream2"
llama_model_id = "meta-llama/Llama-3.2-1B-Instruct"
revision = "2024-08-26"
output_folder = 'vidframes'


# 'latency' wrapper for reporting time spent in executing a function
def latency(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"latency => {func.__name__}: {end_time - start_time:.4f} seconds")
        return result

    return wrapper


def extract_frames(
    video_path: Union[str, os.PathLike],
    output_folder: str = output_folder,
    sample_rate: int = 2,
) -> list:

    os.makedirs(
        output_folder, exist_ok=True
    )  # Create output folder if it doesn't exist

    video = cv2.VideoCapture(video_path)  # read video file with cv2

    # get video properties
    fps = video.get(cv2.CAP_PROP_FPS)  # frames per second in the video
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # total number of frames
    frame_interval = int(
        fps * sample_rate
    )  # Calculate the frame interval based on the sample rate

    # Initialize frame counter
    frame_count = 0

    while True:
        success, frame = video.read()  # Read a frame
        # Extract frame at specified intervals
        if frame_count % frame_interval == 0 and success:
            frame_filename = os.path.join(output_folder, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_filename, frame)  # write frame to jpeg/image file
        else:
            break  # stop when theres an error in frame extraction

        frame_count += 1

    # Release the video capture object
    video.release()

    print(
        f"Extracted {frame_count // frame_interval} of {num_frames} total frames, image frames saved at {output_folder}"
    )
    
    vidframes = [os.path.join(output_folder, path) for path in os.listdir(output_folder)]

    image_frames = [pillow.open(img) for img in vidframes]

    return image_frames


def load_models(
    md_model_path: Union[str, os.PathLike] = local_model_path,
    md_tokenizer_path: Union[str, os.PathLike] = local_tokenizer_path,
    llama_path: str = llama_model_path,
    model_id: str = model_id,
    llama_id: str = llama_model_id,
) -> tuple:
    md_model = None
    md_tokenizer = None
    llama_pipe = None

    is_local = os.path.isdir(
        md_model_path
    )  # check if previously saved models are available
    llm_is_local = os.path.isdir(llama_model_path)

    if is_local and llm_is_local:  # load from locally saved weights
        print("loading from local checkpoint")
        md_model = AutoModelForCausalLM.from_pretrained(md_model_path)
        md_tokenizer = AutoTokenizer.from_pretrained(md_tokenizer_path)
        llama_pipe = pipeline(
            "text-generation",
            model=llama_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    else:  # download fresh weights from huggingface
        print("downloading weights from huggingface")
        md_model = AutoModelForCausalLM.from_pretrained(model_id)
        md_tokenizer = AutoTokenizer.from_pretrained(model_id)
        llama_pipe = pipeline(
            "text-generation",
            model=llama_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        # then save locally for next time
        llama_pipe.save_pretrained(llama_model_path)
        md_model.save_pretrained(md_model_path)
        md_tokenizer.save_pretrained(md_tokenizer_path)  # type: ignore

    return md_model, md_tokenizer, llama_pipe
