# 文件名：main_sim.py (基础运动指令自检版)
import numpy as np
import robosuite as suite
from vla_model import OpenVLAgent
import cv2
import torch

def run_vla_grasping():
    print("初始化 Robosuite 环境 (全局视角 - 基础运动测试)...")
    
    # 使用 agentview，坐标系相对固定：
    # X轴：前后； Y轴：左右； Z轴：上下
    env = suite.make(  
       env_name="PickPlace", 
       robots="Panda",             
       has_renderer=True,          
       has_offscreen_renderer=True, 
       use_camera_obs=True,        
       camera_names="agentview",    
       controller_configs=suite.load_controller_config(default_controller="OSC_POSE"),
       horizon=2000 
    )
      
    agent = OpenVLAgent()
    obs = env.reset()

    # --- 任务序列：强制单一方向指令 ---
    # 我们把词汇量降到最低，去除所有修饰语，只留“动词+方位”
    test_sequence = [
        {"desc": "左移测试", "cmd": "MOVE LEFT", "steps": 8},
        {"desc": "右移测试", "cmd": "MOVE RIGHT", "steps": 8},
        {"desc": "上移测试", "cmd": "MOVE UP", "steps": 8}
    ]

    print("开始基础运动自检。按 'q' 退出。")
    
    try:
        for stage in test_sequence:
            print(f"\n>>> 阶段任务: {stage['desc']} | 核心指令: {stage['cmd']}")
            
            for s in range(stage['steps']):
                # 图像处理
                img = obs["agentview_image"]
                img = np.flipud(img)          
                img_uint8 = (img * 255).astype(np.uint8) 
                img_uint8 = cv2.resize(img_uint8, (224, 224), interpolation=cv2.INTER_AREA)

                # AI 推理
                # 使用全大写，在很多 LLM 底座中，这代表更高权重的指令
                action = agent.predict_action(img_uint8, stage['cmd'])

                # 核心调试打印：我们只看位移
                raw_xyz = action[:3]
                print(f"  {stage['cmd']} {s+1}/{stage['steps']} | 原始位移: {np.round(raw_xyz, 4)}")

                # 维度补位
                if env.action_dim == 8 and len(action) == 7:
                    action = np.append(action, 0.0)
                
                # 动作映射
                action[-1] = 1.0 # 保持夹爪开启，排除干扰
                
                # 【动作缩放】调到 3.5 倍。
                # 如果这个倍数下它还在“原地调整”，说明 4-bit 量化已经把这个方向词的权重删了。
                action[:3] *= 3.5  

                # 执行 (Chunking 10 帧，确保肉眼能观察到趋势)
                for _ in range(10):
                    obs, reward, done, info = env.step(action)
                    env.render()
                    if done: break
                
                # 每步给用户一点观察时间，并检查退出
                if cv2.waitKey(1) & 0xFF == ord('q'): return

        print("\n所有测试阶段已完成。")

    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        env.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_vla_grasping()
