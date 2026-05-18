# Intellifacade
AI-driven interactive facade system. Combines gesture recognition &amp; behavior detection (YOLOv11) for touch-free environmental control. Designed for smart healthcare.
# IntelliFacade: AI-Powered Interactive Smart Facade

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-blueviolet.svg)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)

> **An AI-driven, touch-free, behavior-adaptive facade control system designed for smart healthcare environments.**  
> 本项目是一个面向智慧疗愈场景的 **AI驱动·无接触·行为自适应** 智能立面控制系统。

## 💡 产品设计逻辑 (Product Design)

### 核心痛点与价值
在医院、疗养院等场景中，患者常常行动不便，无法自主调节环境光照；医护人员又难以实时兼顾多床位患者的环境需求。IntelliFacade 通过 AI 视觉技术，实现了**从“人找控制”到“环境自适应”** 的交互范式革新。

### 交互模式设计
系统定义了两种核心交互模式，以应对不同用户状态：
1.  **主动手势控制 (Active Gesture Control)**：面向有行动能力的用户，通过简单的手势运动方向，实现对光照模式的主动调节。
2.  **被动行为响应 (Passive Behavior Adaptation)**：面向行动不便或突发状况的患者，系统根据人体行为（行走、休息、跌倒）**自动、无感**地切换立面模式。

## ⚙️ 系统架构与技术栈 (System Architecture)

### 核心技术栈
| 技术模块 | 技术选型 | 应用说明 |
|---|---|---|
| **手势识别** | OpenCV + MediaPipe | 实现方向性手势判定，内置防抖机制 |
| **行为识别** | YOLOv11 (Ultralytics) | 识别行走(Walking)/休息(Resting)/跌倒(Fall) |
| **边缘推理** | ONNX Runtime / TensorRT | 低延迟边缘计算，满足<0.5s实时响应 |
| **能源系统** | 光伏供电 | 能源自维持，支持不间断运行 |

### 行为-立面映射策略 (Behavior-Facade Logic)
我们设计了一套场景化的映射规则，让立面能够“理解”人的需求：

| 识别状态 (Status) | 立面模式 (Mode) | 透光率 (Transmittance) | 设计意图 (Intent) |
|---|---|---|---|
| 🧍 Walking (行走) | **导航模式** | 渐变引导光 | 提供路径照明，营造安全感 |
| 🛌 Resting (休息) | **隐私模式** | 低透光/柔光 | 保障私密空间，促进身心放松 |
| 🆘 Fall (跌倒) | **紧急模式** | **全透光 + 报警** | 第一时间暴露现场，并触发通知 |

## 🔄 交互体验迭代 (UX Iteration)

我们非常注重交互的自然性与容错率，经历了一次核心交互的跃迁：
*   **V1.0 (特定手势)**：定义了拇指向上/下、五指张开等特定手势。
    *   *发现的问题*：用户学习成本高，误判率高达 `1.3次/任务`。
*   **V2.0 (方向性手势·当前版本)**：简化为手部方向移动（上移/下移/左移/右移），并加入**防抖算法**。
    *   *成果*：极大降低用户记忆负担，**误判率降低 85%**，交互直觉显著提升。

## 📊 项目成果与关键指标 (Key Results)

-   **原型系统**：已完成可演示的集成原型，打通“感知-决策-执行”闭环。
-   **性能指标**：
    -   边缘推理延迟 `< 0.5s`（满足实时交互要求）
    -   跌倒检测准确率 `92%`（基于公开数据集与模拟测试）
-   **产品文档**：已输出完整的产品需求文档 (`/docs/PRD.md`) 与技术选型说明。

## 🛠 快速开始 (Quick Start)

*[此处按需补充你的项目安装、配置和运行步骤]*

1.  **克隆仓库**
    ```bash
    git clone https://github.com/你的用户名/intellifacade.git
    cd intellifacade
