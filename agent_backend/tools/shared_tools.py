"""
shared_tools.py — AgentWithSkills 的元工具 schema 定义。
"""

READ_SKILL_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "read_skill",
        "description": (
            "加载指定 skill 的完整 SOP（Level 1），或其内部支持文件（Level 2）。"
            "决定使用某个 skill 前必须先加载其 SOP，已加载的 skill 无需重复加载。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "skill 名称，如 analyze-network"},
                "path": {
                    "type": "string",
                    "description": "可选。skill 文件夹内的支持文件路径（Level 2）",
                },
            },
            "required": ["name"],
        },
    },
}

BASH_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "执行 bash 命令，通常用于运行 skills/<name>/scripts/*.py 脚本工具。",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "要执行的 bash 命令"},
            },
            "required": ["command"],
        },
    },
}

MODIFY_OUTLINE_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "modify_outline",
        "description": (
            "对当前大纲执行一个或多个修改操作（增删节点、保留分支、修改节点属性等）。"
            "参数经 JSON 传递、完全不过 shell，无需任何引号转义。整棵大纲重建用 set_outline。"
            "\n\n【ops 支持的操作类型】"
            "\n- delete_node    删除节点及其全部子树。必填：node_id"
            "\n- add_node       新增节点，挂到指定父节点下。必填：node_id, parent_id；可选：after_id（插入到此兄弟节点之后）。"
            "node_id 在知识图谱中存在时自动拉取完整子树；不在 KB 中时需额外传 name（和可选 description）创建自定义结构节点"
            "\n- keep_only_node 保留该节点，删除同级所有其他节点。必填：node_id"
            "\n- update_node    修改节点属性值。必填：node_id, field, value。"
            "field 可选：name / description（L5 禁止）/ condition / exec_sql（仅 L5，"
            "写入 sql_config.exec_sql）/ descriptionSuggestion / summarySuggestion / condition_queries。"
            "descriptionSuggestion 是生成报告时 LLM 用来写 description 的格式规则（只呈现数据、不做分析），summarySuggestion 则允许结合数据做分析。"
            "name 改名后 L5 节点自动从 KB 同步 sql_config 等关联字段。"
            "exec_sql 只能在原始 SQL 基础上做局部改动，禁止引入原 SQL 中不存在的字段或枚举值，修改前必须先用 get_node_detail.py 查看当前值。"
            "\n\n多个独立操作可合并为一次调用。"
            "跳过的操作会在返回内容中以 SKIPPED 标注——出现时必须继续补救，不得告知用户已完成。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ops": {
                    "type": "array",
                    "description": "操作数组，每项是一个包含 op 字段的操作对象",
                    "items": {
                        "type": "object",
                        "properties": {
                            "op": {"type": "string",
                                   "description": "操作类型：delete_node / add_node / keep_only_node / update_node"},
                            "node_id": {"type": "string",
                                        "description": "目标节点 ID；自定义节点用 new_L<层级>_<序号> 格式"},
                            "parent_id": {"type": "string",
                                          "description": "（add_node 专用）父节点 ID，填新节点的直接父节点，不能填兄弟节点"},
                            "after_id": {"type": "string",
                                         "description": "（add_node 可选）将新节点插入到此兄弟节点之后；省略则追加到末尾"},
                            "name": {"type": "string",
                                     "description": "（add_node 自定义节点专用）节点名称；node_id 不在知识图谱中时必填"},
                            "description": {"type": "string", "description": "（add_node 自定义节点可选）节点描述"},
                            "field": {"type": "string", "description": "（update_node 专用）要修改的字段名"},
                            "value": {"description": "（update_node 专用）新值；condition_queries 传数组，其余传字符串"},
                        },
                        "required": ["op", "node_id"],
                    },
                },
            },
            "required": ["ops"],
        },
    },
}

SET_OUTLINE_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "set_outline",
        "description": (
            "一次性写入一份完整的报告大纲（结构由你自己组合）。"
            "\n\n【适用场景】用户/专家想自己组合报告结构时使用——典型是专家知识沉淀（把一段业务方法论组织成章节大纲），"
            "或用户明确要求按自定义结构搭建报告。本工具是整棵覆盖写入，会替换当前大纲。"
            "\n【不适用】在已有大纲上做局部改动，用 modify_outline。"
            "\n\n【结构约束】"
            "\n1. outline 是节点对象数组，且恰好一个根节点(L1，报告总标题)"
            "\n2. 根节点的 name 即报告总标题，必须与用户/专家描述中已给出的标题完全一致——若已明确命名，原文照用，不得擅自摘要或改写"
            "\n3. 每个节点是对象 {id, name, description, children}，子节点放进 children 数组（无子节点可省略 children）"
            "\n4. L5 query 节点必须是叶子(无 children)，其 id 须引用 search_graph_tree 返回的知识库已有 id，禁止新建 query 节点"
            "\n5. 新建结构节点 id 必须按 new_L<层级>_<序号> 命名，显式编码层级：根用 new_L1_xxx，"
            "其下依次 new_L2_xxx / new_L3_xxx / new_L4_xxx（结构节点只能 L1~L4，绝不能 L5）"
            "\n6. 新建节点必须填写 description(50~100 字)"
            "\n7. 只写 id/name/description/children(及按需 condition/condition_queries)，不要写 level/sql_config 等字段"
            "\n\n成功时工具会回显写入后的大纲；返回'写入失败'或未回显即为失败，须修正重试，不得告知用户已生成。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "outline": {
                    "type": "array",
                    "description": (
                        "【类型：array，禁止传字符串】"
                        "\n直接传 JSON 数组对象，绝对不能把数组序列化成字符串后传入。"
                        "\n错误示范（会解析失败）：\"outline\": \"[{\\\"id\\\": \\\"new_L1_root\\\", ...}]\""
                        "\n正确示范：\"outline\": [{\"id\": \"new_L1_root\", \"name\": \"报告标题\", \"description\": \"...\","
                        " \"children\": [{\"id\": \"new_L2_001\", \"name\": \"章节\", \"description\": \"...\","
                        " \"children\": [{\"id\": \"L5_001\", \"name\": \"指标名\"}]}]}]"
                        "\n顶层含一个 L1 根节点，每个节点含 id/name/description，子节点放 children 数组。"
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string",
                                   "description": "节点 ID：新建结构节点用 new_L<层级>_<序号>，知识库叶子节点用原 id（如 L5_001）"},
                            "name": {"type": "string",
                                     "description": "节点名称；根节点（L1）的 name 必须与用户/专家描述中已给出的标题完全一致，原文照用"},
                            "description": {"type": "string", "description": "节点描述，新建节点必填，50~100 字"},
                            "children": {"type": "array", "description": "子节点数组，L5 叶子节点不填此字段",
                                         "items": {"type": "object"}},
                            "condition": {"type": "string"},
                            "condition_queries": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["id", "name"],
                    },
                },
            },
            "required": ["outline"],
        },
    },
}
# ── knowledge-fusion 原生工具 ───────────────────────────────────

LIST_TEMPLATES_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "list_templates",
        "description": "按 ID 批量读取知识融合所需的模板元数据和完整大纲。",
        "parameters": {
            "type": "object",
            "properties": {
                "template_ids": {
                    "type": "array", "minItems": 1, "uniqueItems": True,
                    "items": {"type": "string", "minLength": 1},
                    "description": "参与融合的模板 ID 列表",
                },
            },
            "required": ["template_ids"],
            "additionalProperties": False,
        },
    },
}

SHOW_GRAPH_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "show_graph",
        "description": "按知识图谱 ID 返回完整 nodes、relations 和版本。",
        "parameters": {
            "type": "object",
            "properties": {
                "graph_id": {"type": "string", "minLength": 1, "description": "目标知识图谱 ID"},
            },
            "required": ["graph_id"],
            "additionalProperties": False,
        },
    },
}

_GRAPH_CREATE_NODE = {
    "type": "object",
    "properties": {
        "op": {"type": "string", "enum": ["create_node"]},
        "id": {"type": "string", "pattern": r"^new_L[2-4]_[A-Za-z0-9_-]+$"},
        "name": {"type": "string", "minLength": 1},
        "parent_id": {"type": "string", "minLength": 1},
        "description": {"type": "string"},
        "descriptionSuggestion": {"type": "string"},
        "condition": {"type": "string"},
        "condition_queries": {"type": "array", "items": {"type": "string"}},
        "summarySuggestion": {"type": "string"},
        "summary": {"type": "string"},
    },
    "required": ["op", "id", "name"],
    "additionalProperties": False,
}

_GRAPH_ATTACH_SQL_NODE = {
    "type": "object",
    "properties": {
        "op": {"type": "string", "enum": ["attach_sql_node"]},
        "node_id": {"type": "string", "pattern": r"^L5_[A-Za-z0-9_-]+$"},
        "parent_id": {"type": "string", "minLength": 1},
    },
    "required": ["op", "node_id", "parent_id"],
    "additionalProperties": False,
}

_GRAPH_MODIFY_NODE = {
    "type": "object",
    "properties": {
        "op": {"type": "string", "enum": ["modify_node"]},
        "node_id": {"type": "string", "minLength": 1},
        "column_name": {"type": "string", "enum": ["description", "condition", "name"]},
        "new_content": {"type": "string"},
    },
    "required": ["op", "node_id", "column_name", "new_content"],
    "additionalProperties": False,
}

_GRAPH_DELETE_NODE = {
    "type": "object",
    "properties": {
        "op": {"type": "string", "enum": ["delete_node"]},
        "node_id": {"type": "string", "minLength": 1},
    },
    "required": ["op", "node_id"],
    "additionalProperties": False,
}

GRAPH_MANAGE_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "graph_manage",
        "description": (
            "执行用户已确认的知识图谱操作。支持 create_node、attach_sql_node、"
            "modify_node、delete_node；未获得用户确认前禁止调用。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "graph_id": {"type": "string", "minLength": 1},
                "operations": {
                    "type": "array", "minItems": 1,
                    "items": {"anyOf": [
                        _GRAPH_CREATE_NODE,
                        _GRAPH_ATTACH_SQL_NODE,
                        _GRAPH_MODIFY_NODE,
                        _GRAPH_DELETE_NODE,
                    ]},
                },
            },
            "required": ["graph_id", "operations"],
            "additionalProperties": False,
        },
    },
}

KNOWLEDGE_FUSION_TOOLS = [LIST_TEMPLATES_TOOL, SHOW_GRAPH_TOOL, GRAPH_MANAGE_TOOL]
