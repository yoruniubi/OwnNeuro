import json
import os

def recount_motion(motion: dict) -> tuple[int, int, int]:
    """
    重新计算*.motion3.json文件中CurveCount, TotalSegmentCount和TotalPointCount的值
    """
    segment_count = 0
    point_count = 0
    
    # 兼容大小写并检查字段存在性
    curves = motion.get("Curves") or motion.get("curves")
    if not curves:
        raise ValueError("Motion文件缺少 'Curves' 或 'curves' 字段")
    
    curve_count = len(curves)
    for curve in curves:
        segments = curve.get("Segments") or curve.get("segments")
        if not segments:
            continue  # 跳过无效曲线
        
        end_pos = len(segments)
        point_count += 1
        v = 2
        while v < end_pos:
            identifier = segments[v]
            if identifier in (0, 2, 3):
                point_count += 1
                v += 3
            elif identifier == 1:
                point_count += 3
                v += 7
            else:
                raise ValueError(f"未知的标识符: {identifier}")
            segment_count += 1
    return curve_count, segment_count, point_count

def load_all_motion_path_from_model_dir(model_dir: str) -> list[str]:
    """导入模型文件夹中所有有效的motion3.json文件路径"""
    motions_dir = os.path.join(model_dir, "motions")
    if not os.path.exists(motions_dir):
        print(f"[WARN] 目录 '{motions_dir}' 不存在")
        return []
    
    motion_files = []
    for filename in os.listdir(motions_dir):
        if not filename.endswith(".json"):
            continue  # 跳过非JSON文件
        full_path = os.path.join(motions_dir, filename)
        if os.path.isfile(full_path):
            motion_files.append(full_path)
    return motion_files

def load_motion_from_path(path: str) -> dict:
    """安全加载JSON文件"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] 无法加载文件 '{path}': {str(e)}")
        return {}

def copy_modify_from_motion(motion_path: str, save_root: str = "./out/motions") -> None:
    """修复并保存动作文件"""
    try:
        motion = load_motion_from_path(motion_path)
        if not motion:
            return
        
        # 检查必要字段
        if "Meta" not in motion:
            print(f"[WARN] 文件 '{motion_path}' 缺少 'Meta' 字段")
            return
        
        # 重新计算数值
        curve_count, segment_count, point_count = recount_motion(motion)
        motion["Meta"]["CurveCount"] = curve_count
        motion["Meta"]["TotalSegmentCount"] = segment_count
        motion["Meta"]["TotalPointCount"] = point_count
        
        # 确保保存目录存在
        os.makedirs(save_root, exist_ok=True)
        save_path = os.path.join(save_root, os.path.basename(motion_path))
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(motion, f, indent=2, ensure_ascii=False)
        print(f"[INFO] 已修复并保存文件: {save_path}")
    except Exception as e:
        print(f"[ERROR] 处理文件 '{motion_path}' 失败: {str(e)}")

if __name__ == "__main__":
    # 测试用例
    motionPathList = load_all_motion_path_from_model_dir('./models/1009109')
    for path in motionPathList:
        copy_modify_from_motion(path, save_root="./fixed_motions/motions")