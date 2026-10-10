import time
import traceback
from pathlib import Path

import dxcam
from ultralytics import YOLO


# 项目根目录与权重目录：均以脚本自身位置(__file__)为锚点，不依赖运行时 cwd
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# ── 推理参数 ──
CONF_THRESHOLD = 0.4         # 置信度阈值：低于此值的框不绘制
BOX_COLOR      = 0xFF00FF00  # 检测框颜色，格式 0xAARRGGBB（不透明绿）
TEXT_COLOR     = 0xFFFFFFFF  # 标签文字颜色（不透明白）
BOX_THICKNESS  = 2           # 检测框线宽（像素）
TEXT_SIZE      = 16          # 标签字号
TEXT_OFFSET_Y  = 18          # 标签相对框顶部的上移量（像素）
FRAME_INTERVAL = 0.1         # 每帧间隔（秒）


def select_model(models_dir: Path = MODELS_DIR) -> str:
    """扫描 models/ 目录，引导用户交互式选择用于推理的权重文件。

    不猜、不硬编码——模型清单完全来自 models/ 目录下的实际文件。

    Args:
        models_dir: 权重所在目录，默认项目根的 models/
    Returns:
        选中权重的路径（字符串，可直接传给 YOLO()）
    Raises:
        FileNotFoundError: 目录不存在，或目录下没有任何 .pt 权重
    """
    models_dir = Path(models_dir)
    if not models_dir.is_dir():
        raise FileNotFoundError(f"权重目录不存在：{models_dir}")

    models = sorted(models_dir.glob("*.pt"))
    if not models:
        raise FileNotFoundError(f"在 {models_dir} 下没找到任何 .pt 权重文件")

    print(f"\n📦 在 {models_dir.name}/ 下发现 {len(models)} 个权重：")
    for i, p in enumerate(models, 1):
        print(f"   [{i:2d}] {p.name}")

    while True:
        try:
            raw = input(f"\n👉 请选择用于推理的模型 [1-{len(models)}]（回车默认 1）：").strip()
        except EOFError:          # 非交互环境（管道 / 重定向）拿不到输入，走默认
            raw = ""

        if raw == "":
            idx = 0
        elif raw.isdigit() and 1 <= int(raw) <= len(models):
            idx = int(raw) - 1
        else:
            print(f"   [!] 输入无效，请输入 1 ~ {len(models)} 之间的编号。")
            continue

        chosen = models[idx]
        print(f"   ✓ 已选择：{chosen.name}\n")
        return str(chosen)


def load_overlay():
    """导入 overlay 扩展；缺失时先编译，再重试导入。

    Returns:
        overlay 模块
    Raises:
        RuntimeError: 编译失败，或编译完成后仍无法导入
    """
    try:
        from overlay import overlay
        return overlay
    except ModuleNotFoundError:
        pass

    print("未找到 overlay 扩展，正在编译……")
    try:
        from overlay import build
        build.build()
    except Exception:
        raise RuntimeError(f"overlay 编译失败：\n{traceback.format_exc()}")

    try:
        from overlay import overlay
        return overlay
    except Exception:
        raise RuntimeError(f"编译完成但仍无法导入 overlay：\n{traceback.format_exc()}")


def main() -> None:
    try:
        # 1) 准备 overlay 扩展
        try:
            overlay = load_overlay()
        except RuntimeError as e:
            print(e)
            return

        # 2) 选择模型
        try:
            model = YOLO(select_model())
        except Exception:
            print(f"加载模型失败：\n{traceback.format_exc()}")
            return

        # 3) 初始化屏幕捕获
        try:
            camera = dxcam.create()
        except Exception:
            print(f"初始化 dxcam 屏幕捕获失败：\n{traceback.format_exc()}")
            return

        # 4) 创建叠加层并进入检测主循环
        ov = overlay.Overlay()
        try:
            ov.show()

            while True:
                time.sleep(FRAME_INTERVAL)

                # 每帧开头清空并立刻提交一次：让窗口先变全透明，再截图。
                # 否则桌面复制会把上一帧画上去的框当成画面内容喂给 YOLO，形成自反馈。
                ov.clear()
                ov.present()

                capture = camera.grab_view()
                if capture is None:
                    continue

                # conf= 让 ultralytics 在 NMS 阶段就滤掉低分框，不必再手动比较
                result = model.predict(capture, conf=CONF_THRESHOLD, verbose=False)

                for loc in result[0].boxes.data:
                    x1, y1, x2, y2, confidence, class_id = loc
                    ov.draw_box(int(x1), int(y1), int(x2), int(y2), BOX_COLOR, BOX_THICKNESS)
                    ov.draw_text(int(x1), int(y1) - TEXT_OFFSET_Y,
                                 f"{model.names[int(class_id)]} {confidence:.2f}",
                                 TEXT_COLOR, TEXT_SIZE)

                # 一帧只提交一次：present 会把整屏 buffer 交给 DWM，放进内层循环会重复拷贝。
                ov.present()
        except Exception:
            print(f"主循环异常：\n{traceback.format_exc()}")
        finally:
            ov.hide()
            camera.release()
    except KeyboardInterrupt:
        print("\n已退出。")


if __name__ == "__main__":
    main()
