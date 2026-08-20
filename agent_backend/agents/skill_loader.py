#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
skill_loader.py — hermes-agent 风格的三级渐进式 skill 加载。

Level 0  system prompt 注入       → skill name/description（启动时缓存）
Level 1  read_skill(name)         → 完整 SKILL.md 正文（SOP）
Level 2  read_skill(name, path)   → skills/<name>/<path> 指定文件内容
"""

from pathlib import Path
import re
import yaml


_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\s*", re.DOTALL)


def _parse_frontmatter(text: str) -> dict:
    """使用 PyYAML 解析 frontmatter，正确处理 > 折叠标量和嵌套块。"""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}
    try:
        data = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        return {}
    if not isinstance(data, dict):
        return {}

    result: dict = {}
    for k, v in data.items():
        if k == "metadata" and isinstance(v, dict):
            hermes = v.get("hermes", {}) or {}
            for hk, hv in hermes.items():
                result[f"hermes.{hk}"] = hv
        else:
            result[k] = str(v).strip() if v is not None else ""
    return result


def discover_skills(skills_dir: Path) -> list[dict]:
    """
    扫描 skills_dir 下所有 */SKILL.md（支持 category/skill-name/ 两级结构）。
    只读 frontmatter，返回 Level 0 元数据列表。
    """
    skills = []
    for skill_md in sorted(skills_dir.glob("**/SKILL.md")):
        text = skill_md.read_text(encoding="utf-8")
        meta = _parse_frontmatter(text)
        meta.setdefault("name", skill_md.parent.name)
        meta.setdefault("category", skill_md.parent.parent.name
                        if skill_md.parent.parent != skills_dir else "")
        meta["_path"] = skill_md.parent
        skills.append(meta)
    return skills


def read_skill(skill_path: Path, ref_path: str | None = None) -> str:
    """
    Level 1: ref_path=None  → 返回 SKILL.md 正文（frontmatter 之后的部分）
    Level 2: ref_path 指定  → 返回 skill 文件夹内的指定文件内容
    """
    if ref_path:
        target = (skill_path / ref_path).resolve()
        if not target.is_relative_to(skill_path.resolve()):
            return f"路径越界: {ref_path}"
        if not target.exists():
            return f"文件不存在: {ref_path}"
        return target.read_text(encoding="utf-8")

    text = (skill_path / "SKILL.md").read_text(encoding="utf-8")
    m = _FRONTMATTER_RE.match(text)
    return text[m.end():].strip() if m else text.strip()
