""""
local_sanitizer.py-定义local在两个阶段的操作

def  prepare_remote_request   ->构建云端服务请求（FPE sanitized_text+t2）
def apply_remote_dag ->根据云端响应进行第三阶段本地脱敏（还未包含语义泄露）

"""


from typing import Dict, List, Optional, Tuple, Union

from api import NERAPI
from entity_types import is_t1_fpe, is_t1_name, is_t2
from dag_module import T2DAGProcessor
from ff3_module import FPEManager
from name_replacer_module import NameReplacer
from tee_contract import RelationEdgePayload, T2EntityPayload, TEEProcessRequestPayload


def _parse_numeric(token: str) -> Tuple[bool, Union[int, float]]:
    try:
        return True, int(token)
    except ValueError:
        pass
    try:
        return True, float(token)
    except ValueError:
        return False, 0


def _find_entity_positions(prompt: str, ner_result: List[Tuple[str, str]]) -> Dict[int, Tuple[int, int]]:
    positions: Dict[int, Tuple[int, int]] = {}
    cursor = 0
    for i, (token, _label) in enumerate(ner_result):
        start = prompt.find(token, cursor)
        if start == -1:
            start = prompt.find(token)
        if start == -1:
            continue
        end = start + len(token)
        positions[i] = (start, end)
        cursor = end
    return positions


class Vault:
    def __init__(self):
        self._data: Dict[str, Dict] = {}

    def store_t1_name(self, fake_name: str, original: str) -> None:
        self._data[fake_name] = {"type": "t1_name", "original": original}

    def store_t1_fpe(self, enc: str, original: str, enc_meta: List) -> None:
        self._data[enc] = {"type": "t1_fpe", "original": original, "enc_meta": enc_meta}

    def to_dict(self) -> Dict:
        return dict(self._data)


class LocalPreprocessor:
    def __init__(self, epsilon: float, ner: Optional[NERAPI] = None):
        self.epsilon = epsilon
        self.name_replacer = NameReplacer()
        self.fpe_manager = FPEManager()
        self.ner = ner or NERAPI()

    def prepare_remote_request(self, prompt: str) -> Tuple[TEEProcessRequestPayload, Dict]:
        # ========================  刷新密钥，避免跨轮攻击  ====================================
        self.name_replacer.refresh_session()
        self.fpe_manager.refresh_session_tweak()
        # ==================================================================================

        ner_result, _unused_edges = self.ner(prompt) #NER
        token_positions = _find_entity_positions(prompt, ner_result)

        vault = Vault()   #脱敏映射记录表
        fpe_sanitized_text = prompt

        t1_entities = sorted(
            [
                (i, item)
                for i, item in enumerate(ner_result)
                if is_t1_name(item[1]) or is_t1_fpe(item[1])
            ],
            key=lambda x: len(x[1][0]),
            reverse=True,
        )
        # ===============================  对t1分类处理  =============================================
        for _i, (token, label) in t1_entities:
            if is_t1_name(label):
                fake_name = self.name_replacer.replace_name(token)
                vault.store_t1_name(fake_name, token)
                fpe_sanitized_text = fpe_sanitized_text.replace(token, fake_name, 1)
            elif is_t1_fpe(label):
                enc, enc_meta = self.fpe_manager.encrypt_master(token)
                vault.store_t1_fpe(enc, token, enc_meta)
                fpe_sanitized_text = fpe_sanitized_text.replace(token, enc, 1)
        # =======================================================================================
        # ================================ 构建t2  =================================================
        t2_entities: List[T2EntityPayload] = []
        for i, (token, label) in enumerate(ner_result):
            if not is_t2(label):
                continue
            ok, value = _parse_numeric(token)
            if not ok:
                continue
            start_pos, end_pos = token_positions.get(i, (-1, -1))
            t2_entities.append(T2EntityPayload(
                token=token,
                label=label,
                value=float(value),
                start_pos=start_pos,
                end_pos=end_pos,
            ))
        # ================================================================================
        request = TEEProcessRequestPayload(
            fpe_sanitized_text=fpe_sanitized_text,
            t2_entities=t2_entities,
            epsilon=self.epsilon,
        )
        session_info = {
            "name_session": self.name_replacer.get_session_info(),
            "fpe_key_material": self.fpe_manager.get_key_material(),
            "ner_result": ner_result,
            "vault": vault.to_dict(),
        }
        return request, session_info

    def apply_remote_dag(
        self,
        request: TEEProcessRequestPayload,
        relation_edges: List[RelationEdgePayload],
    ) -> Tuple[str, Dict]:
       """
        处理云端请求。进行最后脱敏
        """
       raw_edges = []
       #   ================================================================
       #                   目前是单父节点模式
       #   ================================================================
       for edge in relation_edges:
           parent = request.t2_entities[edge.from_entity_index]
           child = request.t2_entities[edge.to_entity_index]
           raw_edges.append({
               "parent_tokens": parent.token,
               "child_token": child.token,
               "rel_type": edge.relation_type,
               "param": edge.param,
           })
       #  ===================================================================
       t2_list = []
       for idx, entity in enumerate(request.t2_entities):
           ok, value = _parse_numeric(entity.token)
           if ok:
               t2_list.append((idx, entity.token, value))
       #  ===================================================================
       #                          添加语义脱敏过滤部分
       #  ===================================================================
       #
       #  ====================================================================
       processor = T2DAGProcessor(total_epsilon=request.epsilon)
       noisy_map, dag_info = processor.process(t2_list, raw_edges)

       sanitized_text = request.fpe_sanitized_text   #FPE扰动结果
       indexed_entities = sorted(
           enumerate(request.t2_entities),
           key=lambda x: len(x[1].token),
           reverse=True,
       )  # t2也逆序遍历
       for idx, entity in indexed_entities:
           ok, val = _parse_numeric(entity.token)
           if not ok:
               continue
           noisy = noisy_map.get(idx, val)
           if isinstance(val, int) and isinstance(noisy, (int, float)):
               noisy_str = str(int(round(noisy)))
           else:
               noisy_str = str(noisy)
           sanitized_text = sanitized_text.replace(entity.token, noisy_str, 1)

       return sanitized_text, dag_info
