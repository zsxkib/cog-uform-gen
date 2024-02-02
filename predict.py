# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import time
import torch
import subprocess
from PIL import Image
from uform import gen_model
from cog import BasePredictor, Input, Path

UFORM_CACHE_DIR = "./cache"
UFORM_MODEL_NAME = "models--unum-cloud--uform-gen"
UFORM_URL = f"https://weights.replicate.delivery/default/uform/{UFORM_MODEL_NAME}.tar"


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self) -> None:
        model_weights_path = os.path.join(UFORM_CACHE_DIR, UFORM_MODEL_NAME)
        if not os.path.exists(model_weights_path):
            print(f"Downloading model weights to {model_weights_path}...")
            download_weights(UFORM_URL, model_weights_path)

        self.model = gen_model.VLMForCausalLM.from_pretrained(
            "unum-cloud/uform-gen",
            cache_dir=UFORM_CACHE_DIR,
            local_files_only=True,
        )
        self.processor = gen_model.VLMProcessor.from_pretrained(
            "unum-cloud/uform-gen",
            cache_dir=UFORM_CACHE_DIR,
            local_files_only=True,
        )

    def generate_caption(self, image, prompt):
        inputs = self.processor(texts=[prompt], images=[image], return_tensors="pt")
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                do_sample=False,
                use_cache=True,
                max_new_tokens=128,
                eos_token_id=32001,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        decoded_text = self.processor.batch_decode(output[:, prompt_len:])[0]
        return decoded_text.replace("<|im_end|>", "").strip()

    def predict(
        self,
        image: Path = Input(description="Input image"),
        prompt: str = Input(
            description="Prompt to guide the caption generation",
            default="Describe the image in great detail",
        ),
    ) -> str:
        image = Image.open(str(image)).convert("RGB")
        caption = self.generate_caption(image, prompt)
        return caption
