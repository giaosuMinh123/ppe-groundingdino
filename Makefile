PYTHON := python

CONFIG_FILE := groundingdino/config/GroundingDINO_SwinT_OGC.py
CHECKPOINT := D:/grounding_dino/GroundingDINO/weights/groundingdino_swint_ogc.pth
IMAGE_PATH := D:\grounding_dino\GroundingDINO\New_test\images\pexels-photo-209230.jpeg
TEXT_PROMPT := "person, ear, ear-mufs, face, face-guard, face-mask, foot, tool, glasses, gloves, helmet, hands, hea, medical-suit, shoes, safety-suit"
OUTPUT_DIR := D:/grounding_dino/outputs

.PHONY: infer

infer:
	$(PYTHON) demo/inference_on_a_image.py \
		--config_file $(CONFIG_FILE) \
		--checkpoint "$(CHECKPOINT)" \
		--image_path "$(IMAGE_PATH)" \
		--text_prompt "$(TEXT_PROMPT)" \
		--output_dir "$(OUTPUT_DIR)" \
		--cpu-only