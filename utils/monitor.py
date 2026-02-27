import time
import threading
import psutil
import os
from typing import Dict

# 尝试导入 pynvml (nvidia-ml-py)
try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False
    print("[Monitor] Warning: 'nvidia-ml-py' not installed. VRAM monitoring disabled.")

class ResourceMonitor:
    def __init__(self, interval: float = 0.2):
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = None
        self.max_ram_mb = 0.0
        self.max_vram_mb = 0.0
        self._lock = threading.Lock()
        
        self.nvml_initialized = False
        self.baseline_vram_mb = 0.0  # 记录监控开始时的显存底噪

        if HAS_PYNVML:
            try:
                pynvml.nvmlInit()
                self.nvml_initialized = True
                self.device_count = pynvml.nvmlDeviceGetCount()
            except pynvml.NVMLError as e:
                print(f"[Monitor] NVML Init Failed: {e}")

    def _get_current_total_vram(self):
        """获取当前所有 GPU 的总显存占用 (MB)"""
        if not self.nvml_initialized:
            return 0.0
        
        total_used = 0.0
        try:
            for i in range(self.device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_used += mem_info.used
        except pynvml.NVMLError:
            pass
        return total_used / (1024 ** 2)

    def _monitor_loop(self):
        current_process = psutil.Process(os.getpid())

        while not self.stop_event.is_set():
            try:
                # 1. 计算 CPU RAM (RSS)
                # recursive=True 查找子进程
                children = current_process.children(recursive=True)
                all_procs = [current_process] + children
                
                total_ram = 0.0
                for p in all_procs:
                    try:
                        if p.is_running():
                            total_ram += p.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                total_ram_mb = total_ram / (1024 ** 2)

                # 2. 计算 GPU VRAM (全局差值法)
                # 这种方法在 Windows 上最有效：计算 (当前全卡占用 - 起始全卡占用)
                # 它可以捕捉到子进程加载模型引起的显存飙升，且不会因为 PID 查不到而返回 0
                current_global_vram = self._get_current_total_vram()
                
                # 显存增量（如果小于0则记为0，防止释放显存后变成负数）
                current_process_vram_mb = max(0.0, current_global_vram - self.baseline_vram_mb)

                # 3. 更新峰值
                with self._lock:
                    self.max_ram_mb = max(self.max_ram_mb, total_ram_mb)
                    self.max_vram_mb = max(self.max_vram_mb, current_process_vram_mb)

            except Exception as e:
                # 捕获所有异常防止崩溃，只打印一次错误日志防止刷屏
                pass 
            
            time.sleep(self.interval)

    def start(self):
        self.stop_event.clear()
        self.max_ram_mb = 0.0
        self.max_vram_mb = 0.0
        
        # 【关键】记录启动瞬间的显存占用作为“底噪”
        # 这样之后计算的显存就是模型加载带来的“增量”
        if self.nvml_initialized:
            self.baseline_vram_mb = self._get_current_total_vram()
        
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()

    def stop(self) -> Dict[str, float]:
        self.stop_event.set()
        if self.thread:
            self.thread.join()
        
        return {
            "max_ram_mb": self.max_ram_mb,
            "max_vram_mb": self.max_vram_mb
        }

# ================================
# 上下文管理器封装 (不变)
# ================================
class WatermarkProfiler:
    def __init__(self, name="Unknown"):
        self.monitor = ResourceMonitor(interval=0.2)
        self.name = name

    def __enter__(self):
        print(f"--- [Profiler] Start monitoring: {self.name} ---")
        self.monitor.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        stats = self.monitor.stop()
        print(f"--- [Profiler] Result for {self.name} ---")
        print(f"    Max RAM Usage : {stats['max_ram_mb']:.2f} MB")
        print(f"    Max VRAM Usage: {stats['max_vram_mb']:.2f} MB")
        print("-----------------------------------------")