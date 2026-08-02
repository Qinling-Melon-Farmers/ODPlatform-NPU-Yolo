"""桌面端通用 UI 构建辅助（views 层共享）。"""

from __future__ import annotations

from PySide6.QtWidgets import QGroupBox, QHBoxLayout, QPushButton, QVBoxLayout, QWidget


def with_buttons(line_edit, buttons: list[tuple[str, object]]) -> QWidget:
    """输入框右侧附加按钮组。"""
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(line_edit)
    for label, callback in buttons:
        button = QPushButton(label)
        button.clicked.connect(callback)
        layout.addWidget(button)
    return container


def form_page(title: str, rows: list[tuple[str, QWidget]]) -> QWidget:
    """表单参数页（任务启动各任务的 stack 页）。"""
    from PySide6.QtWidgets import QFormLayout

    page = QWidget()
    layout = QVBoxLayout(page)
    group = QGroupBox(title)
    form = QFormLayout(group)
    for label, widget in rows:
        form.addRow(label, widget)
    layout.addWidget(group)
    layout.addStretch(1)
    return page
