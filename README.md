# YOLOv8 动画桌面捕捉软件 v2.0 (模块化组件化版)

本项目是一个使用 `YOLOv8` 实现的单人和多人动画桌面捕捉软件。**该版本彻底重构了底层架构，采用了模块化和组件化的开发模式。**

## 📂 项目结构

```
.
├── main.py                     # PyQt6 GUI应用程序主入口
├── src/
│   ├── core/                   # 核心业务逻辑
│   │   ├── capture_worker.py   # 后台捕捉与AI推理线程
│   │   └── recorder.py         # 独立的视频录制线程
│   ├── gui/                    # 界面逻辑
│   │   └── components/         # 独立UI组件
│   │       ├── preview_panel.py  # 视频预览与录制面板组件
│   │       └── settings_panel.py # 参数设置面板组件
│   └── utils/                  # 通用工具
│       ├── config.py           # 常量与骨骼映射配置
│       ├── drawing.py          # OpenCV 绘图函数
│       └── filters.py          # 一欧元滤波器算法
├── requirements.txt            # 依赖包列表
└── README.md                   # 本文档
```

## ✨ 核心亮点

- **架构重构**: 采用模块化设计，核心数据抓取（`core`）与视图渲染（`gui`）解耦。UI 逻辑进一步拆分成多个独立组件，极大降低了代码耦合度。
- **性能优化**: 彻底将视频录制抽离至独立的 `VideoRecorder` 线程，通过队列进行通信。录制 MP4 时不再会阻塞 YOLO 推理和屏幕截图主循环，杜绝录制掉帧现象。
- **配置持久化**: 通过 `settings.json` 本地记录用户的角色、模型、置信度等选择，重启程序配置不再丢失。
- **交互完善**: 补全了针对背景切换的完整功能支持，且支持直接通过 UI 选择本地图片作为自定义背景并自动刷新。

## 📦 安装指南

1. **环境准备**：建议使用 Python 3.10+
2. **安装依赖**:
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 启动命令

```bash
python main.py
```
