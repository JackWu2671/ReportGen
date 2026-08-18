#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
skill_registry.py — SkillRegistry：skill 的注册、查询、禁用与重载。

职责边界：
  - 管理磁盘上的 skill 元数据（discover、get、search、disable、reload）
  - 提供 read_sop() 读取 SOP 正文（包装 skill_loader.read_skill）
  - 不持有对话状态（已加载到 context 的记录由 AgentWithSkills._loaded 管理）
"""

import logging
from pathlib import Path

from agent_with_skills.skill_loader import discover_skills, read_skill

logger = logging.getLogger(__name__)


class SkillRegistry:
    def __init__(self, skills_dir: Path) -> None:
        self._skills_dir = skills_dir
        self._skills: dict[str, dict] = {}   # name → metadata
        self._disabled: set[str] = set()
        self._load()
        logger.info("[SkillRegistry] loaded %d skills from %s", len(self._skills), skills_dir)

    # ── 属性 ──────────────────────────────────────────────────────

    @property
    def total(self) -> int:
        return len(self._skills)

    # ── 查询 ──────────────────────────────────────────────────────

    def list_all(self) -> list[dict]:
        """返回所有已启用 skill 的 Level 0 元数据列表。"""
        return [m for name, m in self._skills.items() if name not in self._disabled]

    def get(self, name: str) -> dict | None:
        """按名称精确查找，不存在或已禁用返回 None。"""
        meta = self._skills.get(name)
        return meta if meta and name not in self._disabled else None

    def search(self, query: str) -> list[dict]:
        """按 name / description / category 子串搜索（大小写不敏感）。"""
        q = query.lower()
        results = []
        for m in self.list_all():
            if q in m.get("name", "").lower():
                results.append(m)
                continue
            if q in m.get("description", "").lower():
                results.append(m)
                continue
            if q in m.get("category", "").lower():
                results.append(m)
        return results

    # ── 读取内容 ──────────────────────────────────────────────────

    def read_sop(self, name: str, ref_path: str | None = None) -> str:
        """
        Level 1: ref_path=None  → 返回 SKILL.md 正文
        Level 2: ref_path 指定  → 返回 skill 目录内的指定文件
        """
        meta = self._skills.get(name)
        if not meta:
            return f"[SkillRegistry] skill 不存在: {name}"
        return read_skill(meta["_path"], ref_path)

    # ── 管理 ──────────────────────────────────────────────────────

    def disable(self, name: str) -> bool:
        """禁用指定 skill（不从磁盘删除，仅从 list_all / get 中隐藏）。"""
        if name not in self._skills:
            return False
        self._disabled.add(name)
        logger.info("[SkillRegistry] disabled: %s", name)
        return True

    def enable(self, name: str) -> bool:
        """重新启用已禁用的 skill。"""
        self._disabled.discard(name)
        return name in self._skills

    def reload(self) -> None:
        """重新扫描 skills 目录，保留已有的禁用状态。"""
        disabled_before = self._disabled.copy()
        self._load()
        self._disabled = disabled_before & set(self._skills)
        logger.info("[SkillRegistry] reloaded: %d skills, %d disabled", len(self._skills), len(self._disabled))

    # ── 内部 ──────────────────────────────────────────────────────

    def _load(self) -> None:
        skills_list = discover_skills(self._skills_dir)
        self._skills = {m["name"]: m for m in skills_list}
