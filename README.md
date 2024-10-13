### moonvid 🎥🌒

[**kaggle notebook**](https://www.kaggle.com/code/tensorkelechi/moondream-video/)

A hacky **video captioning** framework,
 using a small VLM (**moondream v2**) and text language model (**Llama 3.2-1b**). 

#### method
Uses a **vision language model** to generate **text captions** from single frames of a **video** (which is essesntially a sequence of frames), 
and then a **large langauge model** to merge the several captions into  one **single coherent caption**. \

This isn't an '**SOTA**' model, just an experiment.

#### Acknowledgments
- [**moondream v2**](https://github.com/vikhyat/moondream), lightweight vision language model by [**vikhyat**](https://github.com/vikhyat/)
- Meta AI's [**Llama 3.2-1b-instruct**](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct),
 **small** and capable, instruction-tuned **text language** model