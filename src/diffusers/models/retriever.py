"""
Idea for structure
 Retriever aggregates an Index class and a RetrieverConfig class
 The Index class aggregates a Dataset and RetrieverConfig class
 from_pretrained in the retriever's class, it takes in a huggingface path to a dataset, optional path to an index file+config file in huggingface if there is one
 If an index file is provided, add that index to the dataset.
 If the dataset doesn't have the column embedding or a corresponding index file, in the Index class, the index is computed based on the clip model defined in the config. Then add that to the index of the dataset. This is done in the Index class
 In retrieve we just call the retrieve method in the Index class that gets knn based on the faiss embedding.
 In the save_pretrained method, save index using save_faiss_index. Save this dataset along with config.
 The call method will just call retrieve.
 I'll also have a way to pass the clip model and its components via default arguments.
 Test save_pretrained and from_pretrained methods on new dataset.
"""

from transformers import CLIPModel, CLIPFeatureExtractor, CLIPTokenizer
from datasets import load_dataset, Image, load_dataset_builder, load_from_disk, Dataset
import torch
from typing import Callable, List, Optional, Union
import numpy as np
from ..utils import deprecate, logging
from transformers.models.rag.retrieval_rag import LegacyIndex, CustomHFIndex, CanonicalHFIndex
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
from diffusers.pipelines.rdm.pipeline_rdm import preprocess_images
import os

class IndexConfig:
    def __init__(self, clip_name_or_path="openai/clip-vit-large-patch14", dataset_name="Isamu136/oxford_pets_with_l14_emb", \
                 image_column="image", index_name="embeddings", index_path=None, passages_path=None, dataset_set="train"):
        self.clip_name_or_path = clip_name_or_path
        self.dataset_name = dataset_name
        self.image_column = image_column
        self.index_name = index_name
        self.index_path = index_path
        self.passages_path = passages_path
        self.dataset_set = dataset_set

class Index:
    """
    Each index for a retrieval model is specific to the clip model used and the dataset used.
    """
    def __init__(self, config:IndexConfig, dataset: Dataset):
        self.config = config
        self.dataset = dataset
        self.index_initialized = False
        self.index_name = config.index_name
        self.index_path = config.index_path
        self.init_index()
    def set_index_name(self, index_name:str):
        self.index_name = index_name
    def init_index(self):
        if not self.index_initialized:
            if self.index_path and self.index_name:
                try:
                    self.dataset.load_faiss_index(self.index_name, self.index_path)
                    self.index_initialized = True
                except:
                    logger.info("Index not initialized")
            if self.index_name in self.dataset.features:
                self.dataset.add_faiss_index(column=self.index_name)
                self.index_initialized = True
    def build_index(self, clip_model:CLIPModel=None, feature_extractor:CLIPFeatureExtractor=None, device:str="cuda", torch_dtype=torch.float32):
        if not self.index_initialized:
            clip_model = clip_model or CLIPModel.from_pretrained(self.config.clip_name_or_path).to(device=device, dtype=torch_dtype)
            feature_extractor = feature_extractor or CLIPFeatureExtractor.from_pretrained(self.config.clip_name_or_path)
            self.dataset = get_dataset_with_emb(self.dataset, clip_model, feature_extractor, device=device, image_column=self.config.image_column, index_name=self.config.index_name)
            self.init_index()
    def retrieve_imgs(self, vec, k:int=20):
        vec = np.array(vec).astype(np.float32)
        return self.dataset.get_nearest_examples(self.index_name, vec, k=k)
    def retrieve_embs(self, vec, k:int=20):
        vec = np.array(vec).astype(np.float32)
        return self.dataset.search(self.index_name, vec, k=k)
class Retriever:
    def __init__(self, config:IndexConfig, index:Index=None, dataset:Dataset=None, clip_model:CLIPModel=None,\
                         feature_extractor:CLIPFeatureExtractor=None):
        self.config = config
        self.index = index or self._build_index(config, dataset, clip_model=clip_model, feature_extractor=feature_extractor)
    @classmethod
    def from_pretrained(cls, retriever_name_or_path:str, index:Index=None, dataset:Dataset=None, clip_model:CLIPModel=None,\
                         feature_extractor:CLIPFeatureExtractor=None, **kwargs):
        config = kwargs.pop("config", None) or IndexConfig.from_pretrained(retriever_name_or_path, **kwargs)
        return cls(
            config,
            index=index,
            dataset=dataset,
            clip_model=clip_model,
            feature_extractor=feature_extractor
        )
    @staticmethod
    def _build_index(config:IndexConfig, dataset:Dataset=None, clip_model:CLIPModel=None,\
                         feature_extractor:CLIPFeatureExtractor=None):
        dataset = dataset or load_dataset(config.dataset_name)
        dataset = dataset[config.dataset_set]
        index =Index(config, dataset)
        index.build_index(clip_model=clip_model, feature_extractor=feature_extractor)
        return index

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        if self.config.index_path  is None:
            index_path = os.path.join(save_directory, "hf_dataset_index.faiss")
            self.index.dataset.get_index(self.config.index_name).save(index_path)
            self.config.index_path = index_path
        if self.config.passages_path is None:
            passages_path = os.path.join(save_directory, "hf_dataset")
            # datasets don't support save_to_disk with indexes right now
            faiss_index = self.index.dataset._indexes.pop(self.config.index_name)
            self.index.dataset.save_to_disk(passages_path)
            self.index.dataset._indexes[self.config.index_name] = faiss_index
            self.config.passages_path = passages_path
        self.config.save_pretrained(save_directory)

    def init_retrieval(self):
        """
        Retriever initialization function. It loads the index into memory.
        """

        logger.info("initializing retrieval")
        self.index.init_index()
    def retrieve_imgs(self, embeddings: np.ndarray, k: int):
        """
        Retrieves images for specified `embeddings`.
        Args:
            embeddings (`np.ndarray` of shape `(vector_size)`):
                A batch of query vectors to retrieve with.
            k (`int`):
                The number of nearest neighbor images retrieved per query.
        Return:
            `Tuple[np.ndarray, np.ndarray, List[dict]]`: A tuple with the following objects:
            - **retrieved_doc_embeds** (`np.ndarray` of shape `(batch_size, n_docs, dim)`) -- The retrieval embeddings
              of the retrieved docs per query.
            - **doc_dicts** (`List[dict]`): The `retrieved_doc_embeds` examples per query.
        """
        return self.index.retrieve_imgs(embeddings, k)
    def retrieve_embs(self, embeddings: np.ndarray, k: int):
        """
        Retrieves images for specified `embeddings`.
        Args:
            embeddings (`np.ndarray` of shape `(vector_size)`):
                A batch of query vectors to retrieve with.
            k (`int`):
                The number of nearest neighbor images retrieved per query.
        Return:
            `Tuple[np.ndarray, np.ndarray, List[dict]]`: A tuple with the following objects:
            - **retrieved_doc_embeds** (`np.ndarray` of shape `(batch_size, n_docs, dim)`) -- The retrieval embeddings
              of the retrieved docs per query.
            - **doc_dicts** (`List[dict]`): The `retrieved_doc_embeds` examples per query.
        """
        return self.index.retrieve_embs(embeddings, k)
    def __call__(
        self,
        embeddings,
        k: int=None,
    ):
        return self.index.retrieve_embs(embeddings, k)
def map_img_to_clip_feature(clip, feature_extractor, imgs, device="cuda"):
    for i, image in enumerate(imgs):
        if not image.mode == "RGB":
            imgs[i] = image.convert("RGB")
    retrieved_images = preprocess_images(imgs, feature_extractor).to(device)
    image_embeddings = clip.get_image_features(retrieved_images)
    image_embeddings = image_embeddings / torch.linalg.norm(image_embeddings, dim=-1, keepdim=True)
    image_embeddings = image_embeddings[None, ...]
    return image_embeddings
def map_txt_to_clip_feature(clip, tokenizer, prompt, device="cuda"):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids

    if text_input_ids.shape[-1] > tokenizer.model_max_length:
        removed_text = tokenizer.batch_decode(text_input_ids[:, tokenizer.model_max_length :])
        logger.warning(
            "The following part of your input was truncated because CLIP can only handle sequences up to"
            f" {tokenizer.model_max_length} tokens: {removed_text}"
        )
        text_input_ids = text_input_ids[:, :tokenizer.model_max_length]
    text_embeddings = clip.get_text_features(text_input_ids.to(device))
    text_embeddings = text_embeddings / torch.linalg.norm(text_embeddings, dim=-1, keepdim=True)
    text_embeddings = text_embeddings[:, None, :]
    return text_embeddings[0][0].cpu().detach().numpy()
def get_dataset_with_emb(dataset, clip_model, feature_extractor, device="cuda", image_column="image", index_name="embeddings"):
    return dataset.map(lambda example: {index_name: map_img_to_clip_feature(clip_model, feature_extractor, [example[image_column]], device).cpu().detach().numpy()[0][0]})