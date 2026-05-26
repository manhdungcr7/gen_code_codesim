import subprocess
import os
import time

def run_in_sandbox(ai_code: str, test_cases: str, timeout: int = 15):
    """
    Thực thi mã nguồn trong Docker Sandbox của BigCodeBench.
    Trả về: (boolean_thành_công, string_log_lỗi)
    """
    # 1. Ghép code của AI và test cases lại thành 1 script hoàn chỉnh
    full_code = f"{ai_code}\n\n# --- TEST CASES ---\n{test_cases}"
    
    # 2. Tạo file tạm trên máy
    current_dir = os.path.abspath(os.getcwd())
    temp_file_name = "sandbox_temp.py"
    temp_file_path = os.path.join(current_dir, temp_file_name)
    
    with open(temp_file_path, "w", encoding="utf-8") as f:
        f.write(full_code)

    # 3. Cấu hình lệnh gọi Docker gọi vào file tạm
    docker_cmd = [
        "docker", "run", "--rm",
        "--entrypoint", "timeout", 
        "-v", f"{current_dir}:/app",
        "-w", "/app",
        "bigcodebench/bigcodebench-evaluate:latest",
        str(timeout), "python", temp_file_name
    ]

    # 4. Gửi lệnh vào Docker và bắt kết quả (Log)
    try:
        result = subprocess.run(docker_cmd, capture_output=True, text=True, timeout=timeout+5)
        
        # Thêm trễ 2.0s để Docker kịp dọn vì GPT-4o-mini sinh code rất nhanh
        time.sleep(2.0)
        
        # Dọn dẹp file tạm ngay khi chạy xong
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            
        if result.returncode == 0:
            return True, "Success"
        else:
            error_log = result.stderr if result.stderr else result.stdout
            return False, error_log

    except subprocess.TimeoutExpired:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return False, "TimeoutError: Code execution exceeded the time limit."
    except Exception as e:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return False, f"Sandbox System Error: {str(e)}"