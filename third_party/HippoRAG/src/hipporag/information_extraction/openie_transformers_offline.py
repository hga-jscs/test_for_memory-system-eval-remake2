import json
from typing import Dict, Tuple

from ..information_extraction import OpenIE
from .openie_openai import ChunkInfo
from ..utils.misc_utils import NerRawOutput, TripleRawOutput
from ..utils.logging_utils import get_logger
from ..prompts import PromptTemplateManager

logger = get_logger(__name__)


class TransformersOfflineOpenIE(OpenIE):
    """
    Transformers 离线 OpenIE

    说明：
    - 该模式依赖 transformers/torch 等本地推理环境
    - 这里做了懒加载：只有真的实例化该类时才 import transformers_offline
    """

    def __init__(self, global_config):
        self.prompt_template_manager = PromptTemplateManager(
            role_mapping={"system": "system", "user": "user", "assistant": "assistant"}
        )

        try:
            from ..llm.transformers_offline import TransformersOffline  # type: ignore
        except Exception as e:
            raise ImportError(
                "TransformersOfflineOpenIE 需要 transformers/torch 等依赖，但当前环境导入失败。"
                "如果你不打算用 offline/transformers 模式，可以忽略；否则请安装对应依赖。"
            ) from e

        self.llm_model = TransformersOffline(global_config)

    def batch_openie(self, chunks: Dict[str, ChunkInfo]) -> Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
        """
        Conduct batch OpenIE synchronously using transformers offline batch mode, including NER and triple extraction
        """
        # Extract passages from the provided chunks
        chunk_passages = {chunk_key: chunk["content"] for chunk_key, chunk in chunks.items()}

        ner_input_messages = [self.prompt_template_manager.render(name='ner', passage=p) for p in chunk_passages.values()]
        ner_output, _ner_output_metadata = self.llm_model.batch_infer(ner_input_messages, json_template='ner', max_tokens=512)

        triple_extract_input_messages = [
            self.prompt_template_manager.render(
                name='triple_extraction',
                passage=passage,
                named_entity_json=named_entities
            )
            for passage, named_entities in zip(chunk_passages.values(), ner_output)
        ]
        triple_output, _triple_output_metadata = self.llm_model.batch_infer(triple_extract_input_messages, json_template='triples', max_tokens=2048)

        ner_raw_outputs = []
        for idx, ner_output_instance in enumerate(ner_output):
            chunk_id = list(chunks.keys())[idx]
            response = ner_output_instance
            try:
                unique_entities = json.loads(response).get("named_entities", [])
            except Exception as e:
                unique_entities = []
                logger.warning(f"Could not parse NER response from OpenIE: {e}")
            if len(unique_entities) == 0:
                logger.warning("No entities extracted for chunk_id: %s", chunk_id)
            ner_raw_output = NerRawOutput(chunk_id, response, unique_entities, {})
            ner_raw_outputs.append(ner_raw_output)
        ner_results_dict = {chunk_key: ner_raw_output for chunk_key, ner_raw_output in zip(chunks.keys(), ner_raw_outputs)}

        triple_raw_outputs = []
        for idx, triple_output_instance in enumerate(triple_output):
            chunk_id = list(chunks.keys())[idx]
            response = triple_output_instance
            try:
                triples = json.loads(response).get("triples", [])
            except Exception as e:
                triples = []
                logger.warning(f"Could not parse triple response from OpenIE: {e}")
            if len(triples) == 0:
                logger.warning("No triples extracted for chunk_id: %s", chunk_id)
            triple_raw_output = TripleRawOutput(chunk_id, response, triples, {})
            triple_raw_outputs.append(triple_raw_output)
        triple_results_dict = {chunk_key: triple_raw_output for chunk_key, triple_raw_output in zip(chunks.keys(), triple_raw_outputs)}

        return ner_results_dict, triple_results_dict