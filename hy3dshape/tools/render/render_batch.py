#!/usr/bin/env python3
import os, sys, glob, argparse
import time
import datetime
from types import SimpleNamespace

# 确保可以导入同目录下的 render.py
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if CUR_DIR not in sys.path:
    sys.path.append(CUR_DIR)
from render import main as render_main  # noqa: E402

# 进度条（无 tqdm 时降级为直传 iterator）
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    def tqdm(iterable, total=None, desc=None):  # type: ignore
        return iterable


def parse_args():
    # 兼容 blender -P xxx.py -- 之后的参数
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
    else:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description="Render multiple models one by one")
    parser.add_argument("--input_dir", type=str, required=True, help="包含 glb/obj 等模型的目录")
    parser.add_argument("--out_root", type=str, required=True, help="输出根目录；每个模型将写入 out_root/<stem>/render_cond/")
    parser.add_argument("--patterns", type=str, nargs="*", default=["*.glb", "*.gltf", "*.obj"], help="匹配的通配符列表")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--engine", type=str, default="CYCLES")
    parser.add_argument("--views", type=int, default=24)
    parser.add_argument("--geo_mode", action="store_true", help="是否以几何模式渲染（会导出 mesh.ply）")
    parser.add_argument("--limit", type=int, default=-1, help="最多处理多少个文件，-1 表示全部")
    parser.add_argument("--name_filter", type=str, default="", help="只处理文件名包含指定字符串的文件，如 '_thin'")
    parser.add_argument("--progress_file", type=str, default="", help="进度文件路径，用于存储渲染进度信息")
    parser.add_argument("--file_list", type=str, default="", help="指定文件列表路径，如果提供则只处理列表中的文件")
    args = parser.parse_args(argv)
    return args


def collect_inputs(input_dir: str, patterns, name_filter: str = "", file_list: str = ""):
    files = []
    
    # 如果指定了文件列表，直接从列表读取
    if file_list and os.path.exists(file_list):
        with open(file_list, 'r') as f:
            files = [line.strip() for line in f if line.strip()]
        print(f"[INFO] Loaded {len(files)} files from file list: {file_list}")
    else:
        # 否则从目录中搜索
        for pat in patterns:
            files.extend(glob.glob(os.path.join(input_dir, pat)))
        files = sorted(set(files))
    
    # 如果指定了名称过滤器，只保留包含该字符串的文件
    if name_filter:
        files = [f for f in files if name_filter in os.path.basename(f)]
    
    return files


def make_single_args(batch_args, model_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    # render.py 需要的参数集合
    return SimpleNamespace(
        views=batch_args.views,
        object=model_path,
        output_folder=out_dir,
        resolution=batch_args.resolution,
        engine=batch_args.engine,
        geo_mode=batch_args.geo_mode,
        # 非 geo_mode 下哪些通道默认开启由 render.py 内部设置，这里先置 False
        save_depth=False,
        save_normal=False,
        save_albedo=False,
        save_mr=False,
        save_mist=False,
        split_normal=False,
        save_mesh=False,
    )


def update_progress_file(progress_file: str, current: int, total: int, start_time: float, 
                        current_file: str, status: str = "processing", 
                        resolution: int = None, views: int = None, engine: str = None, geo_mode: bool = None):
    """更新进度文件"""
    if not progress_file:
        return
    
    elapsed_time = time.time() - start_time
    progress_percent = (current / total) * 100 if total > 0 else 0
    
    # 计算预计完成时间
    if current > 0:
        avg_time_per_file = elapsed_time / current
        remaining_files = total - current
        estimated_remaining_time = avg_time_per_file * remaining_files
        estimated_completion = datetime.datetime.now() + datetime.timedelta(seconds=estimated_remaining_time)
    else:
        estimated_remaining_time = 0
        estimated_completion = None
    
    # 创建进度条
    bar_length = 50
    filled_length = int(bar_length * current // total) if total > 0 else 0
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    
    # 格式化时间
    def format_time(seconds):
        if seconds < 60:
            return f"{seconds:.1f}秒"
        elif seconds < 3600:
            return f"{seconds/60:.1f}分钟"
        else:
            return f"{seconds/3600:.1f}小时"
    
    # 生成进度内容
    progress_content = f"""# 渲染进度报告

## 总体进度
- **当前进度**: {current}/{total} ({progress_percent:.1f}%)
- **进度条**: [{bar}] {progress_percent:.1f}%
- **已用时间**: {format_time(elapsed_time)}
- **预计剩余时间**: {format_time(estimated_remaining_time) if estimated_remaining_time > 0 else '计算中...'}
- **预计完成时间**: {estimated_completion.strftime('%Y-%m-%d %H:%M:%S') if estimated_completion else '计算中...'}

## 当前状态
- **正在处理**: {os.path.basename(current_file)}
- **状态**: {status}
- **开始时间**: {datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}
- **最后更新**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 渲染参数
- **分辨率**: {os.environ.get('RESOLUTION', '4096')}
- **视图数**: {os.environ.get('VIEWS', '24')}
- **渲染引擎**: {os.environ.get('ENGINE', 'CYCLES')}
- **几何模式**: {'是' if '--geo_mode' in sys.argv else '否'}

---
*此文件由 render_batch.py 自动生成和更新*
"""
    
    try:
        with open(progress_file, 'w', encoding='utf-8') as f:
            f.write(progress_content)
    except Exception as e:
        print(f"[WARN] 无法更新进度文件 {progress_file}: {e}")


def print_progress_summary(current: int, total: int, start_time: float, current_file: str):
    """打印进度摘要"""
    elapsed_time = time.time() - start_time
    progress_percent = (current / total) * 100 if total > 0 else 0
    
    if current > 0:
        avg_time_per_file = elapsed_time / current
        remaining_files = total - current
        estimated_remaining_time = avg_time_per_file * remaining_files
        estimated_completion = datetime.datetime.now() + datetime.timedelta(seconds=estimated_remaining_time)
        
        print(f"[进度] {current}/{total} ({progress_percent:.1f}%) | "
              f"已用: {elapsed_time/60:.1f}分钟 | "
              f"预计剩余: {estimated_remaining_time/60:.1f}分钟 | "
              f"预计完成: {estimated_completion.strftime('%H:%M:%S')}")
    else:
        print(f"[进度] {current}/{total} ({progress_percent:.1f}%) | 已用: {elapsed_time/60:.1f}分钟")
    
    print(f"[当前] 正在处理: {os.path.basename(current_file)}")


def main():
    args = parse_args()
    inputs = collect_inputs(args.input_dir, args.patterns, args.name_filter, args.file_list)

    # 限制数量
    if args.limit > 0:
        inputs = inputs[:args.limit]

    total = len(inputs)
    print(f"[INFO] Found {total} models to render under {args.input_dir}")
    os.makedirs(args.out_root, exist_ok=True)
    
    # 设置进度文件路径
    if not args.progress_file:
        # 默认在输出目录下创建进度文件
        args.progress_file = os.path.join(args.out_root, "render_progress.md")
    
    # 记录开始时间
    start_time = time.time()
    
    # 初始化进度文件
    update_progress_file(args.progress_file, 0, total, start_time, "", "准备开始")
    print(f"[INFO] 进度文件: {args.progress_file}")

    # 简单的循环处理，一次一个模型
    for i, model_path in enumerate(inputs):
        stem = os.path.splitext(os.path.basename(model_path))[0]
        out_dir = os.path.join(args.out_root, stem, "render_cond")
        
        # 更新进度
        update_progress_file(args.progress_file, i, total, start_time, model_path, "正在渲染")
        print_progress_summary(i, total, start_time, model_path)
        
        print(f"[{i+1}/{total}] Rendering: {stem}")
        
        try:
            single_args = make_single_args(args, model_path, out_dir)
            render_main(single_args)
            print(f"[{i+1}/{total}] ✓ Completed: {stem}")
            
            # 更新进度为完成
            update_progress_file(args.progress_file, i+1, total, start_time, model_path, "已完成")
            
        except Exception as e:
            print(f"[{i+1}/{total}] ✗ Failed: {stem} - {str(e)}")
            # 更新进度为失败
            update_progress_file(args.progress_file, i+1, total, start_time, model_path, f"失败: {str(e)}")
            continue

    # 最终完成状态
    total_time = time.time() - start_time
    update_progress_file(args.progress_file, total, total, start_time, "", "全部完成")
    
    print(f"[DONE] All models processed in {total_time/60:.1f} minutes.")
    print(f"[INFO] 最终进度报告: {args.progress_file}")


if __name__ == "__main__":
    main() 