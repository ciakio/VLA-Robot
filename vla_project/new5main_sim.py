# 文件名：main_sim.py (夹爪指令极致自检版)
import numpy as np
import robosuite as suite
from vla_model import OpenVLAgent
import cv2
import torch

def run_vla_grasping():
    print("初始化 Robosuite 环境 (进行夹爪指令测试)...")
    
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

    # --- 任务序列：强制交替执行“开”和“关” ---
    gripper_sequence = [
        {"desc": "张开指令测试", "cmd": "OPEN THE GRIPPER", "steps": 5},
        {"desc": "合拢指令测试", "cmd": "CLOSE THE GRIPPER", "steps": 5}
    ]

    print("开始夹爪闭环自检。监控第 7 维输出。按 'q' 退出。")
    
    try:
        # 重复 3 次大循环，观察规律
        for cycle in range(3):
            print(f"\n--- 第 {cycle+1} 轮交替测试 ---")
            
            for stage in gripper_sequence:
                print(f"\n>>> 正在下达: {stage['desc']} ({stage['cmd']})")
                
                for s in range(stage['steps']):
                    # 图像处理（虽然是测夹爪，但 VLA 必须输入图像）
                    img = np.flipud(obs["agentview_image"])
                    img_uint8 = (img * 255).astype(np.uint8)
                    img_uint8 = cv2.resize(img_uint8, (224, 224))

                    # AI 推理
                    action = agent.predict_action(img_uint8, stage['cmd'])

                    # --- 【核心观察区】：只看最后一位 ---
                    raw_gripper_val = action[-1]
                    # 提示：OpenVLA 原始输出通常 1.0 是开，0.0 是关
                    print(f"  步数 {s+1} | 指令: {stage['cmd']} | AI原始夹爪值: {round(raw_gripper_val, 4)}")

                    # 动作处理与维度补全
                    if env.action_dim == 8 and len(action) == 7:
                        action = np.append(action, 0.0)
                    
                    # 强行屏蔽 XYZ 的位移，防止机械臂乱飞，我们只看夹爪动不动
                    action[:3] = 0 
                    
                    # 映射夹爪：VLA(0~1) -> Robosuite(-1~1)
                    # 如果 AI 输出 1.0 -> env 1.0 (开)
                    # 如果 AI 输出 0.0 -> env -1.0 (关)
                    action[6] = 2.0 * (raw_gripper_val - 0.5) 

                    # 执行并在窗口观察
                    for _ in range(15):
                        obs, reward, done, info = env.step(action)
                        env.render()
                        if done: break
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'): return

        print("\n夹爪自检序列完成。")

    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        env.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_vla_grasping()
