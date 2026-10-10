"""
YOLO 端到端 Benchmark 工具
功能: 速度测试 + 显存监控 + 精度评估 + 自动选型推荐
"""
import time
import torch

import numpy as np
import yaml

from pathlib import Path
from ultralytics import YOLO
from tabulate import tabulate


# 项目根目录与权重目录：均以脚本自身位置(__file__)为锚点，不依赖运行时 cwd
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


class YOLOBenchmark:
    def __init__(self, models: list[str], data_yaml: str | None = None,
                 image_path: str | None = None, device: str = "0") -> None:
        """
        Args:
            models: 模型路径列表, 如 ["models/yolo11n.pt", "models/yolo11s.pt", "models/yolo11m.pt"]
            data_yaml: 验证集yaml路径(可选, 提供则同时测精度)
            image_path: 测速用图片路径(可选, 不提供则自动生成640x640假图)
            device: "0"=GPU0, "cpu"=CPU
        """
        self.models = models
        self.data_yaml = data_yaml
        self.image_path = image_path
        self.device = device
        self.is_gpu = device != "cpu" and torch.cuda.is_available()
        self.results = []

    # ---------------- 模型发现 ----------------
    @staticmethod
    def discover_models(models_dir: str | Path = MODELS_DIR,
                        pattern: str = "*.pt",
                        exclude: tuple[str, ...] = ()) -> list[str]:
        """扫描目录下所有匹配的权重文件。

        Args:
            models_dir: 权重所在目录（默认项目根的 models/）
            pattern: 文件名通配符，默认 "*.pt"
            exclude: 需排除的文件名子串，例如 ("world", "_cs2")
        Returns:
            按文件名排序后的权重路径列表
        Raises:
            FileNotFoundError: 目录不存在，或过滤后没剩下任何权重
        """
        models_dir = Path(models_dir)
        if not models_dir.is_dir():
            raise FileNotFoundError(f"权重目录不存在: {models_dir}")

        files = sorted(models_dir.glob(pattern))
        if exclude:
            files = [p for p in files if not any(k in p.name for k in exclude)]

        if not files:
            raise FileNotFoundError(
                f"在 {models_dir} 下没找到可用权重（pattern={pattern!r}, exclude={exclude!r}）"
            )
        return [str(p) for p in files]

    @classmethod
    def from_models_dir(cls, models_dir: str | Path = MODELS_DIR,
                        pattern: str = "*.pt", exclude: tuple[str, ...] = (),
                        **kwargs) -> "YOLOBenchmark":
        """从目录自动发现权重并构造实例。

        用法：
            bench = YOLOBenchmark.from_models_dir(
                data_yaml="...", image_path="...", device="0")
        """
        models = cls.discover_models(models_dir, pattern, exclude)
        return cls(models=models, **kwargs)

    # ---------------- 输入校验 ----------------
    # 数据集配置必须齐全的字段：任何一个为空（null / 空串 / 空容器）都不算可用
    REQUIRED_DATA_FIELDS = ("path", "train", "val", "nc", "names")

    @staticmethod
    def _is_blank(value) -> bool:
        """None / 空串 / 空容器都算"空"。注意 0 和 False 不算空。"""
        if value is None:
            return True
        if isinstance(value, str):
            return value.strip() == ""
        if isinstance(value, (list, tuple, dict, set)):
            return len(value) == 0
        return False

    @classmethod
    def _data_yaml_problem(cls, path: str | Path) -> str | None:
        """检查数据集配置是否可用。

        Returns:
            问题描述；配置完好则返回 None
        """
        p = Path(path)
        if not p.exists():
            return "文件不存在"
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8"))
        except Exception as e:
            return f"解析失败（{type(e).__name__}: {e}）"

        if not isinstance(data, dict):
            return "内容不是有效的键值结构"

        blank = [k for k in cls.REQUIRED_DATA_FIELDS if cls._is_blank(data.get(k))]
        if blank:
            return f"以下字段为空：{', '.join(blank)}"
        return None

    # 测速图候选后缀
    IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")

    def _dataset_image_dir(self) -> Path | None:
        """从 data_yaml 的 path + train/val 推断出图片目录；推不出来返回 None。"""
        if not self.data_yaml:
            return None
        try:
            data = yaml.safe_load(Path(self.data_yaml).read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(data, dict):
            return None

        cfg_dir = Path(self.data_yaml).resolve().parent
        root = Path(data.get("path") or cfg_dir)
        if not root.is_absolute():
            root = (cfg_dir / root).resolve()

        split = data.get("val") or data.get("train")
        if isinstance(split, (list, tuple)):
            split = split[0] if split else None
        if not isinstance(split, str) or not split.strip():
            return None

        d = Path(split)
        return d if d.is_absolute() else root / d

    def _resolve_image_path(self) -> None:
        """未指定测速图时，从数据集的图片目录里自动挑一张。

        挑不到就把 image_path 保持为 None —— _benchmark_speed 会退化成随机图测速。
        """
        def fallback(reason: str, hint: str = "") -> None:
            print(f"[i] {reason}")
            print("    将改用随机生成的图片测速（延迟 / 显存仍具参考价值）。")
            if hint:
                print(f"    {hint}")

        img_dir = self._dataset_image_dir()
        if img_dir is None:
            fallback("未指定测速图，也无法从数据集配置推断出图片目录。")
            return
        if not img_dir.is_dir():
            fallback(f"图片目录不存在：{img_dir}",
                     f"提示：把测试图片放进 {img_dir} 即可用真实图片测速。")
            return

        candidates = sorted(p for ext in self.IMAGE_EXTS for p in img_dir.glob(ext))
        if not candidates:
            fallback(f"图片目录里没有图片：{img_dir}",
                     f"提示：把测试图片放进 {img_dir} 即可用真实图片测速。")
            return

        self.image_path = str(candidates[0])
        print(f"[i] 未指定测速图，自动选用：{candidates[0].name}")

    def _check_inputs(self) -> None:
        """校验 data_yaml / image_path；不可用就降级并提示，避免每个模型都失败一遍。

        - data_yaml 除了要存在，字段还必须齐全——留空的模板（如未填值的 data.yaml）
          会让精度测试报错，这里直接判定为不可用并跳过。
        - image_path 没指定（或指向的文件不存在）时，自动从数据集图片目录里挑一张；
          连图片都没有就置为 None，由 _benchmark_speed 生成随机图。
        """
        if self.data_yaml:
            problem = self._data_yaml_problem(self.data_yaml)
            if problem:
                print(f"[!] 数据集配置不可用：{self.data_yaml}")
                print(f"    原因：{problem}")
                print("    已跳过精度测试（只测速度）。")
                print("    提示：把 test/data.yaml 复制为 test/data_local.yaml 并填入真实路径。")
                self.data_yaml = None

        if self.image_path and not Path(self.image_path).exists():
            print(f"[!] 指定的测速图片不存在：{self.image_path}")
            self.image_path = None
        if not self.image_path:
            self._resolve_image_path()

    def _benchmark_speed(self, model_path: str, num_warmup: int = 20,
                         num_runs: int = 100) -> dict:
        """端到端速度 + 显存测试"""
        model = YOLO(model_path)
        
        # 准备测试图片
        img = self.image_path if self.image_path else np.random.randint(
            0, 255, (640, 640, 3), dtype=np.uint8
        )
        
        # 重置显存统计
        if self.is_gpu:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        # 预热
        for _ in range(num_warmup):
            model.predict(img, verbose=False, device=self.device)
        if self.is_gpu:
            torch.cuda.synchronize()
        
        # 正式测速
        latencies = []
        for _ in range(num_runs):
            if self.is_gpu:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.predict(img, verbose=False, device=self.device)
            if self.is_gpu:
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)
        
        latencies = np.array(latencies)
        peak_vram = torch.cuda.max_memory_allocated() / (1024**2) if self.is_gpu else 0
        
        return {
            "avg_ms": float(np.mean(latencies)),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "fps": float(1000 / np.mean(latencies)),
            "vram_mb": float(peak_vram),
        }

    def _benchmark_accuracy(self, model_path: str) -> dict:
        """精度测试 (mAP)"""
        if not self.data_yaml:
            return {"map50": None, "map50_95": None}
        model = YOLO(model_path)
        metrics = model.val(data=self.data_yaml, batch=1, device=self.device, verbose=False)
        return {
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
        }

    def run(self) -> list[dict]:
        """执行完整 Benchmark"""
        self._check_inputs()

        print(f"\n🚀 开始 Benchmark | 设备: {'GPU:'+self.device if self.is_gpu else 'CPU'}")
        print(f"   模型数量: {len(self.models)} | 精度测试: {'✅ '+self.data_yaml if self.data_yaml else '❌ 跳过'}")
        for m in self.models:
            print(f"     • {Path(m).name}")
        print()

        for i, m in enumerate(self.models, 1):
            print(f"[{i}/{len(self.models)}] ⏳ 测试 {m} ...")
            try:
                speed = self._benchmark_speed(m)
                acc = self._benchmark_accuracy(m)
                self.results.append({"model": m, **speed, **acc})
                line = f"         ✅ {speed['avg_ms']:.1f}ms | FPS:{speed['fps']:.0f}"
                # 用 is not None 判断：mAP50 恰好为 0.0 也是有效值，不能被当成"没测"
                if acc["map50"] is not None:
                    line += f" | mAP50:{acc['map50']:.3f}"
                print(line)
            except Exception as e:
                print(f"         ❌ 失败: {type(e).__name__}: {e}")

        self._print_report()
        self._recommend()
        return self.results

    def _print_report(self) -> None:
        """打印对比表格"""
        if not self.results:
            return
        headers = ["模型", "Avg(ms)", "P95(ms)", "FPS", "VRAM(MB)", "mAP@50", "mAP@50-95"]
        rows = []
        for r in self.results:
            rows.append([
                r["model"],
                f"{r['avg_ms']:.2f}",
                f"{r['p95_ms']:.2f}",
                f"{r['fps']:.0f}",
                f"{r['vram_mb']:.0f}",
                f"{r['map50']:.3f}" if r['map50'] is not None else "-",
                f"{r['map50_95']:.3f}" if r['map50_95'] is not None else "-",
            ])
        print("\n" + "="*75)
        print("📊 Benchmark 对比报告")
        print("="*75)
        print(tabulate(rows, headers=headers, tablefmt="simple_grid"))

    def _recommend(self) -> None:
        """帕累托最优自动推荐"""
        valid = [r for r in self.results if r.get("map50") is not None]
        if len(valid) < 2:
            print("\n💡 提示: 提供 data_yaml 参数可同时评估精度，获得更精准的选型推荐。")
            if self.results:
                best = min(self.results, key=lambda x: x["avg_ms"])
                print(f"⚡ 仅按速度推荐: {best['model']} ({best['avg_ms']:.1f}ms, {best['fps']:.0f} FPS)")
            return
        
        # 找帕累托前沿: 不存在另一个模型既更快又更准
        pareto = []
        for r in valid:
            dominated = False
            for other in valid:
                if other is r:
                    continue
                if other["avg_ms"] <= r["avg_ms"] and other["map50"] >= r["map50"]:
                    if other["avg_ms"] < r["avg_ms"] or other["map50"] > r["map50"]:
                        dominated = True
                        break
            if not dominated:
                pareto.append(r)
        
        pareto.sort(key=lambda x: x["avg_ms"])
        
        print("\n🏆 帕累托最优模型 (无劣势模型):")
        for p in pareto:
            print(f"   • {p['model']:20s} | {p['avg_ms']:6.1f}ms | mAP50: {p['map50']:.3f}")
        
        # 性价比推荐: mAP增益/延迟代价 最高的拐点
        if len(pareto) >= 2:
            best_ratio = -1
            best_model = pareto[0]
            for i in range(1, len(pareto)):
                delta_map = pareto[i]["map50"] - pareto[i-1]["map50"]
                delta_ms = pareto[i]["avg_ms"] - pareto[i-1]["avg_ms"]
                if delta_ms > 0:
                    ratio = delta_map / delta_ms
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_model = pareto[i]
            print(f"\n⭐ 综合性价比推荐: {best_model['model']}")
            print(f"   理由: 在帕累托前沿中，每毫秒延迟换取的精度增益最高")
        else:
            print(f"\n⭐ 推荐: {pareto[0]['model']} (唯一帕累托最优)")


if __name__ == "__main__":
    bench = YOLOBenchmark.from_models_dir(
        data_yaml=str(BASE_DIR / "test" / "data_local.yaml"),   # ← 数据集yaml, 或设为None只测速度
        # image_path 不指定：自动从数据集的图片目录里挑一张；一张都没有就用随机图
        device="0",                   # ← GPU设备号或"cpu"
        # exclude=("world", "_cs2"),  # ← 如需排除开集模型 / CS2 微调模型，取消注释
    )
    results = bench.run()
