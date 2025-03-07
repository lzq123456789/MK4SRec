import torch
import json
import argparse
from pathlib import Path
from collections import OrderedDict


def convert_tensor_to_serializable(tensor):
    """
    将PyTorch张量转换为可序列化的格式
    """
    if isinstance(tensor, torch.Tensor):
        return {
            'dtype': str(tensor.dtype),
            'shape': list(tensor.shape),
            'data': tensor.numpy().tolist(),
        }
    return tensor


def pt_to_txt(pt_path, output_path=None):
    """
    将PyTorch .pt文件转换为可读的文本格式

    参数:
        pt_path (str): .pt文件的路径
        output_path (str, optional): 输出文本文件的路径
    """
    try:
        # 加载.pt文件
        model_data = torch.load(pt_path, map_location='cpu')

        # 如果未指定输出路径，使用输入文件名
        if output_path is None:
            output_path = str(Path(pt_path).with_suffix('.txt'))

        # 创建用于存储模型信息的字典
        model_info = OrderedDict()

        # 如果是状态字典（state_dict）
        if isinstance(model_data, OrderedDict):
            for key, value in model_data.items():
                model_info[key] = convert_tensor_to_serializable(value)
        # 如果是完整模型
        elif hasattr(model_data, 'state_dict'):
            # 获取模型结构
            model_info['model_structure'] = str(model_data)
            # 获取状态字典
            state_dict = model_data.state_dict()
            model_info['state_dict'] = {}
            for key, value in state_dict.items():
                model_info['state_dict'][key] = convert_tensor_to_serializable(value)
        else:
            model_info = model_data

        # 将模型信息写入文本文件
        with open(output_path, 'w', encoding='utf-8') as f:
            # 写入基本信息
            f.write("PyTorch Model Information\n")
            f.write("=" * 50 + "\n\n")

            # 如果有模型结构信息，先写入结构
            if 'model_structure' in model_info:
                f.write("Model Structure:\n")
                f.write("-" * 20 + "\n")
                f.write(str(model_info['model_structure']))
                f.write("\n\n")

            # 写入详细的参数信息
            f.write("Model Parameters:\n")
            f.write("-" * 20 + "\n")
            json.dump(model_info, f, indent=2, default=str)

        print(f"转换成功！文本文件已保存到: {output_path}")

        # 输出一些基本统计信息
        if isinstance(model_data, OrderedDict):
            num_parameters = len(model_data)
            print(f"模型包含 {num_parameters} 个参数层")

    except Exception as e:
        print(f"转换过程中出现错误: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description='将PyTorch .pt文件转换为可读文本格式')
    parser.add_argument('input_file', help='输入的.pt文件路径')
    parser.add_argument('-o', '--output', help='输出的文本文件路径（可选）')

    args = parser.parse_args()
    pt_to_txt(args.input_file, args.output)


if __name__ == "__main__":
    main()