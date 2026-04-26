from dataclasses import asdict, dataclass
from typing import Dict, List


@dataclass
class T2EntityPayload:
    token: str
    label: str
    value: float
    start_pos: int
    end_pos: int

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TEEProcessRequestPayload:
    fpe_sanitized_text: str
    t2_entities: List[T2EntityPayload]
    epsilon: float

    def to_dict(self) -> Dict:
        return {
            "fpe_sanitized_text": self.fpe_sanitized_text,
            "t2_entities": [entity.to_dict() for entity in self.t2_entities],
            "epsilon": self.epsilon,
        }


@dataclass
class RelationEdgePayload:
    from_entity_index: int
    to_entity_index: int
    relation_type: str
    param: float
    has_temp_node: bool = False
    temp_node_name: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)
