import os
import shutil
import sys

def find_and_copy_sdk():
    print("正在搜索海康 MVS SDK Python 库...")
    
    # 常见的安装路径
    possible_paths = [
        r"C:\Program Files (x86)\MVS\Development\Samples\Python\MvImport",
        r"C:\Program Files\MVS\Development\Samples\Python\MvImport",
        r"D:\Program Files (x86)\MVS\Development\Samples\Python\MvImport",
        r"D:\MVS\Development\Samples\Python\MvImport"
    ]
    
    found_path = None
    for path in possible_paths:
        if os.path.exists(path):
            found_path = path
            break
            
    if not found_path:
        # 尝试深度搜索 (可能会慢)
        print("标准路径未找到，正在尝试搜索安装目录...")
        # 这里简单化，如果没找到就提示用户
        print("\n[错误] 未能自动找到海康 SDK 的 MvImport 文件夹。")
        print("请手动操作：")
        print("1. 找到你的 MVS 安装目录 (通常在 MVS/Development/Samples/Python/MvImport)")
        print("2. 把整个 'MvImport' 文件夹复制到当前代码运行的目录下。")
        return False
        
    print(f"找到 SDK 路径: {found_path}")
    
    target_dir = os.path.join(os.getcwd(), "MvImport")
    if os.path.exists(target_dir):
        print("当前目录下已存在 MvImport，跳过复制。")
        return True
        
    try:
        shutil.copytree(found_path, target_dir)
        print("成功！已将 MvImport 复制到当前目录。")
        return True
    except Exception as e:
        print(f"复制失败: {e}")
        return False

if __name__ == "__main__":
    if find_and_copy_sdk():
        print("\n环境配置完成！现在你可以运行 v3 版本的服务器了。")
        input("按回车键退出...")
    else:
        input("环境配置失败，请按上述提示手动复制...")
